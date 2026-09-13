package com.argent.mobile.llama;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.Settings;
import android.util.Log;

import androidx.core.content.ContextCompat;

import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;

import java.io.File;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

@CapacitorPlugin(name = "ArgentLlama")
public class ArgentLlamaPlugin extends Plugin {
    private static final String TAG = "ArgentLlamaPlugin";

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private boolean isModelLoaded = false;
    private String currentLoadedPath = "";

    static {
        try {
            System.loadLibrary("argent_llama");
            Log.i(TAG, "Библиотека libargent_llama.so успешно загружена");
        } catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "Ошибка загрузки библиотеки libargent_llama: " + e.getMessage());
        }
    }

    public interface TokenCallback {
        void onToken(String token);
    }

    private native boolean nativeLoadModel(String path, int threads, int ctx);
    private native boolean nativeGenerate(String prompt, float temperature, int maxTokens, TokenCallback callback);
    private native void nativeStop();
    private native void nativeUnloadModel();

    @PluginMethod
    public void checkStoragePermission(PluginCall call) {
        boolean granted = false;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            granted = Environment.isExternalStorageManager();
        } else {
            granted = ContextCompat.checkSelfPermission(getContext(),
                    Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED;
        }

        JSObject ret = new JSObject();
        ret.put("granted", granted);
        call.resolve(ret);
    }

    @PluginMethod
    public void requestStoragePermission(PluginCall call) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            try {
                Intent intent = new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION);
                intent.addCategory("android.intent.category.DEFAULT");
                intent.setData(Uri.parse(String.format("package:%s", getContext().getPackageName())));
                getContext().startActivity(intent);
            } catch (Exception e) {
                Intent intent = new Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION);
                getContext().startActivity(intent);
            }
        }
        JSObject ret = new JSObject();
        ret.put("opened", true);
        call.resolve(ret);
    }

    @PluginMethod
    public void loadModel(PluginCall call) {
        String path = call.getString("path");
        if (path == null || path.isEmpty()) {
            call.reject("Путь к файлу модели GGUF не указан");
            return;
        }

        File file = new File(path);
        if (!file.exists()) {
            // Проверяем типовые папки: /storage/emulated/0/Download/ или /sdcard/Download/
            File inDownloads = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), file.getName());
            if (inDownloads.exists()) {
                path = inDownloads.getAbsolutePath();
            } else {
                call.reject("Файл модели не найден по пути: " + path);
                return;
            }
        }

        final String finalPath = path;
        final int threads = call.getInt("threads", 4);
        final int context = call.getInt("context", 2048);

        if (isModelLoaded && finalPath.equals(currentLoadedPath)) {
            JSObject ret = new JSObject();
            ret.put("success", true);
            ret.put("reused", true);
            ret.put("path", finalPath);
            call.resolve(ret);
            return;
        }

        executor.execute(() -> {
            try {
                boolean ok = nativeLoadModel(finalPath, threads, context);
                if (ok) {
                    isModelLoaded = true;
                    currentLoadedPath = finalPath;
                    JSObject ret = new JSObject();
                    ret.put("success", true);
                    ret.put("path", finalPath);
                    call.resolve(ret);
                } else {
                    isModelLoaded = false;
                    call.reject("Не удалось загрузить модель GGUF в нативный движок llama.cpp");
                }
            } catch (Exception e) {
                Log.e(TAG, "Исключение при загрузке модели: " + e.getMessage());
                call.reject("Ошибка загрузки модели: " + e.getMessage());
            }
        });
    }

    @PluginMethod
    public void generateStream(PluginCall call) {
        String prompt = call.getString("prompt", "");
        float temperature = call.getFloat("temperature", 0.6f);
        int maxTokens = call.getInt("max_tokens", 1024);

        if (!isModelLoaded) {
            call.reject("Модель GGUF не загружена в память перед генерацией");
            return;
        }

        executor.execute(() -> {
            try {
                TokenCallback callback = token -> {
                    JSObject data = new JSObject();
                    data.put("token", token);
                    notifyListeners("token", data);
                };

                boolean ok = nativeGenerate(prompt, temperature, maxTokens, callback);
                JSObject ret = new JSObject();
                ret.put("success", ok);
                call.resolve(ret);
            } catch (Exception e) {
                Log.e(TAG, "Ошибка генерации: " + e.getMessage());
                call.reject("Ошибка генерации: " + e.getMessage());
            }
        });
    }

    @PluginMethod
    public void stopGeneration(PluginCall call) {
        try {
            nativeStop();
            JSObject ret = new JSObject();
            ret.put("stopped", true);
            call.resolve(ret);
        } catch (Exception e) {
            call.reject("Ошибка остановки: " + e.getMessage());
        }
    }

    @PluginMethod
    public void unloadModel(PluginCall call) {
        executor.execute(() -> {
            try {
                nativeUnloadModel();
                isModelLoaded = false;
                currentLoadedPath = "";
                JSObject ret = new JSObject();
                ret.put("unloaded", true);
                call.resolve(ret);
            } catch (Exception e) {
                call.reject("Ошибка выгрузки: " + e.getMessage());
            }
        });
    }
}
