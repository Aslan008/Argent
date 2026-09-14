#include <jni.h>
#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <android/log.h>
#include "llama.h"

#define TAG "ArgentLlamaNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static struct llama_model * g_model = nullptr;
static struct llama_context * g_ctx = nullptr;
static const struct llama_vocab * g_vocab = nullptr;
static std::mutex g_mutex;
static std::atomic<bool> g_stop{false};

// Определяет длину префикса в строке, состоящего из полных, валидных UTF-8 символов.
// Если в конце буфера осталась неполная многобайтовая последовательность (например, 1 байт кириллицы из 2-х),
// функция возвращает позицию перед ней, чтобы не передавать неполный UTF-8 в env->NewStringUTF.
static size_t get_valid_utf8_prefix_len(const std::string & str) {
    if (str.empty()) return 0;
    size_t i = 0;
    size_t last_valid = 0;
    const size_t len = str.length();

    while (i < len) {
        unsigned char c = static_cast<unsigned char>(str[i]);
        size_t char_len = 0;
        if ((c & 0x80) == 0x00) {
            char_len = 1; // 1-byte ASCII
        } else if ((c & 0xE0) == 0xC0) {
            char_len = 2; // 2-byte UTF-8 (кириллица и др.)
        } else if ((c & 0xF0) == 0xE0) {
            char_len = 3; // 3-byte UTF-8 (азиатские языки и др.)
        } else if ((c & 0xF8) == 0xF0) {
            char_len = 4; // 4-byte UTF-8 (эмодзи и редкие символы)
        } else {
            i++;
            last_valid = i;
            continue;
        }

        if (i + char_len <= len) {
            bool valid = true;
            for (size_t k = 1; k < char_len; ++k) {
                if ((static_cast<unsigned char>(str[i + k]) & 0xC0) != 0x80) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                i += char_len;
                last_valid = i;
            } else {
                i++;
                last_valid = i;
            }
        } else {
            // Неполная последовательность на границе буфера
            break;
        }
    }
    return last_valid;
}

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_argent_mobile_llama_ArgentLlamaPlugin_nativeLoadModel(
        JNIEnv * env,
        jobject /* thiz */,
        jstring jModelPath,
        jint nThreads,
        jint nCtx) {

    std::lock_guard<std::mutex> lock(g_mutex);

    // Обязательная инициализация бэкендов llama.cpp (ARM NEON, CPU)
    llama_backend_init();

    const char * model_path = env->GetStringUTFChars(jModelPath, nullptr);
    if (!model_path) {
        LOGE("Не удалось получить путь к модели из Java");
        return JNI_FALSE;
    }

    LOGI("Загрузка модели GGUF: %s, потоков: %d, контекст: %d", model_path, nThreads, nCtx);

    // Очищаем предыдущую модель, если была загружена
    if (g_ctx) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
        g_vocab = nullptr;
    }

    // Инициализация параметров модели (mmap включен по умолчанию)
    struct llama_model_params model_params = llama_model_default_params();

    g_model = llama_model_load_from_file(model_path, model_params);
    env->ReleaseStringUTFChars(jModelPath, model_path);

    if (!g_model) {
        LOGE("Ошибка: llama_model_load_from_file вернул null");
        return JNI_FALSE;
    }

    g_vocab = llama_model_get_vocab(g_model);
    if (!g_vocab) {
        LOGE("Ошибка: llama_model_get_vocab вернул null");
        llama_model_free(g_model);
        g_model = nullptr;
        return JNI_FALSE;
    }

    // Инициализация параметров контекста
    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = nCtx > 0 ? nCtx : 2048;
    ctx_params.n_batch = ctx_params.n_ctx;
    ctx_params.n_ubatch = 512;
    ctx_params.n_threads = nThreads > 0 ? nThreads : 2;
    ctx_params.n_threads_batch = ctx_params.n_threads;

    g_ctx = llama_init_from_model(g_model, ctx_params);
    if (!g_ctx) {
        LOGE("Ошибка: llama_init_from_model вернул null");
        llama_model_free(g_model);
        g_model = nullptr;
        g_vocab = nullptr;
        return JNI_FALSE;
    }

    LOGI("Модель GGUF успешно загружена в память устройства!");
    return JNI_TRUE;
}

JNIEXPORT jint JNICALL
Java_com_argent_mobile_llama_ArgentLlamaPlugin_nativeGenerate(
        JNIEnv * env,
        jobject /* thiz */,
        jstring jPrompt,
        jfloat temperature,
        jint maxTokens,
        jobject jCallback) {

    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_model || !g_ctx || !g_vocab) {
        LOGE("Модель не загружена перед генерацией");
        return -1;
    }

    const char * prompt_text = env->GetStringUTFChars(jPrompt, nullptr);
    if (!prompt_text) return -1;

    g_stop = false;

    // Токенизация промпта
    std::string prompt_str(prompt_text);
    env->ReleaseStringUTFChars(jPrompt, prompt_text);

    int n_prompt_tokens = -llama_tokenize(g_vocab, prompt_str.c_str(), prompt_str.length(), nullptr, 0, true, true);
    if (n_prompt_tokens <= 0) {
        n_prompt_tokens = prompt_str.length() + 64;
    }
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    n_prompt_tokens = llama_tokenize(g_vocab, prompt_str.c_str(), prompt_str.length(), prompt_tokens.data(), prompt_tokens.size(), true, true);
    if (n_prompt_tokens < 0) {
        LOGE("Ошибка токенизации промпта");
        return -1;
    }
    prompt_tokens.resize(n_prompt_tokens);

    LOGI("Промпт успешно токенизирован: %zu токенов", prompt_tokens.size());

    // Очищаем состояние памяти/KV-кэша перед новым запросом
    llama_memory_clear(llama_get_memory(g_ctx), true);

    // Инициализация цепочки сэмплеров
    struct llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature > 0.0f ? temperature : 0.6f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(1234));

    // Поиск метода onToken в объекте callback
    jclass callbackClass = env->GetObjectClass(jCallback);
    jmethodID onTokenMethod = env->GetMethodID(callbackClass, "onToken", "(Ljava/lang/String;)V");
    if (!onTokenMethod) {
        LOGE("Не найден метод onToken(String) в переданном callback");
        llama_sampler_free(smpl);
        return -1;
    }

    llama_token new_token_id = LLAMA_TOKEN_NULL;
    struct llama_batch batch = llama_batch_get_one(prompt_tokens.data(), (int)prompt_tokens.size());

    int n_generated = 0;
    int limit = maxTokens > 0 ? maxTokens : 1024;

    LOGI("Старт цикла авторегрессионной генерации (макс токенов: %d)...", limit);

    // Буфер накопления байтов UTF-8 для безопасной передачи кириллицы и CJK в Java
    std::string utf8_accum;
    auto flush_utf8 = [&](bool force) {
        if (utf8_accum.empty()) return;
        size_t valid_len = force ? utf8_accum.length() : get_valid_utf8_prefix_len(utf8_accum);
        if (valid_len > 0) {
            std::string to_send = utf8_accum.substr(0, valid_len);
            utf8_accum.erase(0, valid_len);
            jstring jPiece = env->NewStringUTF(to_send.c_str());
            if (jPiece) {
                env->CallVoidMethod(jCallback, onTokenMethod, jPiece);
                env->DeleteLocalRef(jPiece);
                if (env->ExceptionCheck()) {
                    LOGE("Исключение в Java callback onToken");
                    env->ExceptionDescribe();
                    env->ExceptionClear();
                }
            }
        }
    };

    // Цикл пошаговой генерации токенов
    while (n_generated < limit && !g_stop.load()) {
        if (llama_decode(g_ctx, batch) != 0) {
            LOGE("Ошибка llama_decode на шаге %d", n_generated);
            break;
        }

        new_token_id = llama_sampler_sample(smpl, g_ctx, -1);
        llama_sampler_accept(smpl, new_token_id);

        if (new_token_id == LLAMA_TOKEN_NULL) {
            LOGE("llama_sampler_sample вернул LLAMA_TOKEN_NULL на шаге %d", n_generated);
            break;
        }

        // Проверка на конец генерации (EOS / EOG)
        if (llama_vocab_is_eog(g_vocab, new_token_id)) {
            LOGI("Встречен токен конца ответа EOG (%d). Генерация завершена.", new_token_id);
            break;
        }

        // Преобразование токена в строку с флагом special = true (для тегов рассуждений <think>, </think>)
        char piece_buf[256];
        int n_piece = llama_token_to_piece(g_vocab, new_token_id, piece_buf, sizeof(piece_buf), 0, true);
        if (n_piece > 0) {
            std::string piece_str(piece_buf, n_piece);
            if (piece_str.find("<|im_end|>") != std::string::npos ||
                piece_str.find("<|endoftext|>") != std::string::npos) {
                LOGI("Встречен маркер завершения диалога %s. Остановка генерации.", piece_str.c_str());
                break;
            }
            utf8_accum.append(piece_buf, n_piece);
            flush_utf8(false);
        } else if (n_piece < 0) {
            std::vector<char> big_buf(-n_piece);
            int n_big = llama_token_to_piece(g_vocab, new_token_id, big_buf.data(), big_buf.size(), 0, true);
            if (n_big > 0) {
                std::string piece_str(big_buf.data(), n_big);
                if (piece_str.find("<|im_end|>") != std::string::npos ||
                    piece_str.find("<|endoftext|>") != std::string::npos) {
                    LOGI("Встречен маркер завершения диалога %s. Остановка генерации.", piece_str.c_str());
                    break;
                }
                utf8_accum.append(big_buf.data(), n_big);
                flush_utf8(false);
            }
        }

        // Подготовка пакета для следующего токена
        batch = llama_batch_get_one(&new_token_id, 1);
        n_generated++;

        if (n_generated % 20 == 0) {
            LOGI("Сгенерировано токенов: %d", n_generated);
        }
    }

    // Сбрасываем остаток буфера в UI
    flush_utf8(true);
    llama_sampler_free(smpl);

    LOGI("Генерация завершена. Успешно сгенерировано токенов: %d", n_generated);
    return n_generated;
}

JNIEXPORT void JNICALL
Java_com_argent_mobile_llama_ArgentLlamaPlugin_nativeStop(
        JNIEnv * /* env */,
        jobject /* thiz */) {
    LOGI("Получен сигнал остановки генерации nativeStop");
    g_stop = true;
}

JNIEXPORT void JNICALL
Java_com_argent_mobile_llama_ArgentLlamaPlugin_nativeUnloadModel(
        JNIEnv * /* env */,
        jobject /* thiz */) {
    std::lock_guard<std::mutex> lock(g_mutex);
    LOGI("Выгрузка модели из памяти nativeUnloadModel");
    g_stop = true;
    if (g_ctx) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
        g_vocab = nullptr;
    }
}

} // extern "C"
