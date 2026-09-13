#include <jni.h>
#include <string>
#include <vector>
#include <atomic>
#include <mutex>
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

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_argent_mobile_llama_ArgentLlamaPlugin_nativeLoadModel(
        JNIEnv * env,
        jobject /* thiz */,
        jstring jModelPath,
        jint nThreads,
        jint nCtx) {

    std::lock_guard<std::mutex> lock(g_mutex);

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

    // Инициализация параметров контекста
    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = nCtx > 0 ? nCtx : 2048;
    ctx_params.n_threads = nThreads > 0 ? nThreads : 4;
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

JNIEXPORT jboolean JNICALL
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
        return JNI_FALSE;
    }

    const char * prompt_text = env->GetStringUTFChars(jPrompt, nullptr);
    if (!prompt_text) return JNI_FALSE;

    g_stop = false;

    // Токенизация промпта
    std::string prompt_str(prompt_text);
    env->ReleaseStringUTFChars(jPrompt, prompt_text);

    int n_prompt_tokens = -llama_tokenize(g_vocab, prompt_str.c_str(), prompt_str.length(), nullptr, 0, true, true);
    if (n_prompt_tokens <= 0) {
        n_prompt_tokens = prompt_str.length() + 32;
    }
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    n_prompt_tokens = llama_tokenize(g_vocab, prompt_str.c_str(), prompt_str.length(), prompt_tokens.data(), prompt_tokens.size(), true, true);
    if (n_prompt_tokens < 0) {
        LOGE("Ошибка токенизации промпта");
        return JNI_FALSE;
    }
    prompt_tokens.resize(n_prompt_tokens);

    // Очищаем состояние памяти/KV-кэша перед новым запросом
    llama_memory_clear(llama_get_memory(g_ctx), true);

    // Инициализация сэмплера
    struct llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature > 0.0f ? temperature : 0.6f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(1234));

    // Подача промпта в контекст пакетами (batch)
    const int batch_size = 512;
    for (int i = 0; i < (int)prompt_tokens.size(); i += batch_size) {
        if (g_stop.load()) {
            llama_sampler_free(smpl);
            return JNI_TRUE;
        }
        int cur_batch = std::min(batch_size, (int)prompt_tokens.size() - i);
        struct llama_batch batch = llama_batch_get_one(&prompt_tokens[i], cur_batch);
        if (llama_decode(g_ctx, batch) != 0) {
            LOGE("Ошибка при декодировании промпта в llama_decode");
            llama_sampler_free(smpl);
            return JNI_FALSE;
        }
    }

    // Поиск метода onToken в объекте callback
    jclass callbackClass = env->GetObjectClass(jCallback);
    jmethodID onTokenMethod = env->GetMethodID(callbackClass, "onToken", "(Ljava/lang/String;)V");
    if (!onTokenMethod) {
        LOGE("Не найден метод onToken(String) в переданном callback");
        llama_sampler_free(smpl);
        return JNI_FALSE;
    }

    int n_generated = 0;
    int limit = maxTokens > 0 ? maxTokens : 1024;

    // Цикл пошаговой генерации токенов
    while (n_generated < limit && !g_stop.load()) {
        const llama_token id = llama_sampler_sample(smpl, g_ctx, -1);
        llama_sampler_accept(smpl, id);

        // Проверка на конец генерации (EOS / EOG)
        if (llama_vocab_is_eog(g_vocab, id)) {
            break;
        }

        // Преобразование токена в строку
        char piece_buf[256];
        int n_piece = llama_token_to_piece(g_vocab, id, piece_buf, sizeof(piece_buf), 0, false);
        if (n_piece > 0) {
            std::string piece_str(piece_buf, n_piece);
            jstring jPiece = env->NewStringUTF(piece_str.c_str());
            env->CallVoidMethod(jCallback, onTokenMethod, jPiece);
            env->DeleteLocalRef(jPiece);
        }

        // Подготовка к декодированию следующего токена
        llama_token next_token = id;
        struct llama_batch next_batch = llama_batch_get_one(&next_token, 1);
        if (llama_decode(g_ctx, next_batch) != 0) {
            LOGE("Ошибка декодирования следующего токена");
            break;
        }

        n_generated++;
    }

    llama_sampler_free(smpl);
    LOGI("Генерация завершена. Сгенерировано токенов: %d", n_generated);
    return JNI_TRUE;
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
