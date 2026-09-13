package com.argent.mobile.llama;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.Settings;
import android.util.Log;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.getcapacitor.JSArray;
import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
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
    private native int nativeGenerate(String prompt, float temperature, int maxTokens, TokenCallback callback);
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
        } else {
            if (getActivity() != null) {
                ActivityCompat.requestPermissions(getActivity(),
                        new String[]{Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE},
                        1001);
            }
        }
        JSObject ret = new JSObject();
        ret.put("opened", true);
        call.resolve(ret);
    }

    /**
     * Поиск файла модели по имени или пути в различных каталогах смартфона
     */
    private File resolveModelFile(String path) {
        if (path == null || path.trim().isEmpty()) return null;

        File direct = new File(path);
        if (direct.exists() && direct.isFile()) {
            return direct;
        }

        String targetName = direct.getName().trim();

        List<File> searchDirs = new ArrayList<>();
        File ext = Environment.getExternalStorageDirectory();
        File dl = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS);
        File docs = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS);

        if (dl != null) searchDirs.add(dl);
        if (docs != null) searchDirs.add(docs);
        if (ext != null) {
            searchDirs.add(new File(ext, "Download"));
            searchDirs.add(new File(ext, "Downloads"));
            searchDirs.add(new File(ext, "Download/Telegram"));
            searchDirs.add(new File(ext, "Telegram/Telegram Documents"));
            searchDirs.add(new File(ext, "Documents"));
            searchDirs.add(new File(ext, "Document"));
            searchDirs.add(new File(ext, "Models"));
            searchDirs.add(new File(ext, "LLM"));
            searchDirs.add(ext);
        }
        searchDirs.add(new File("/sdcard/Download"));
        searchDirs.add(new File("/sdcard/Downloads"));
        searchDirs.add(new File("/sdcard"));

        // 1. Точное совпадение имени
        for (File dir : searchDirs) {
            if (dir != null && dir.exists() && dir.isDirectory()) {
                File candidate = new File(dir, targetName);
                if (candidate.exists() && candidate.isFile()) {
                    return candidate;
                }
            }
        }

        // 2. Регистронезависимое совпадение
        for (File dir : searchDirs) {
            if (dir != null && dir.exists() && dir.isDirectory()) {
                File[] list = dir.listFiles();
                if (list != null) {
                    for (File f : list) {
                        if (f.isFile() && f.getName().equalsIgnoreCase(targetName)) {
                            return f;
                        }
                    }
                }
            }
        }

        // 3. Частичное совпадение по корню имени без расширения
        String baseName = targetName.toLowerCase();
        if (baseName.endsWith(".gguf")) baseName = baseName.substring(0, baseName.length() - 5);
        if (baseName.endsWith(".bin")) baseName = baseName.substring(0, baseName.length() - 4);
        
        if (baseName.length() >= 3) {
            for (File dir : searchDirs) {
                if (dir != null && dir.exists() && dir.isDirectory()) {
                    File[] list = dir.listFiles();
                    if (list != null) {
                        for (File f : list) {
                            String fName = f.getName().toLowerCase();
                            if (f.isFile() && (fName.endsWith(".gguf") || fName.endsWith(".bin")) && fName.contains(baseName)) {
                                return f;
                            }
                        }
                    }
                }
            }
        }

        // 4. Поиск в глубину до 3 уровней
        if (ext != null && ext.exists()) {
            File found = findRecursive(ext, targetName, 0, 3);
            if (found != null) return found;
        }

        // 5. Если в папке Download есть единственный .gguf файл — используем его
        if (dl != null && dl.exists()) {
            File[] ggufs = dl.listFiles((d, name) -> name.toLowerCase().endsWith(".gguf"));
            if (ggufs != null && ggufs.length == 1) {
                return ggufs[0];
            }
        }

        return null;
    }

    private File findRecursive(File root, String targetName, int depth, int maxDepth) {
        if (depth > maxDepth || root == null || !root.exists() || !root.isDirectory()) return null;
        File[] list = root.listFiles();
        if (list == null) return null;

        for (File f : list) {
            if (f.isFile() && f.getName().equalsIgnoreCase(targetName)) {
                return f;
            }
        }

        for (File f : list) {
            if (f.isDirectory()) {
                String name = f.getName();
                if (name.startsWith(".") || name.equals("Android") || name.equals("DCIM") ||
                    name.equals("Pictures") || name.equals("Music") || name.equals("Alarms")) {
                    continue;
                }
                File found = findRecursive(f, targetName, depth + 1, maxDepth);
                if (found != null) return found;
            }
        }
        return null;
    }

    @PluginMethod
    public void resolvePath(PluginCall call) {
        String inputPath = call.getString("path");
        File resolved = resolveModelFile(inputPath);
        JSObject ret = new JSObject();
        if (resolved != null && resolved.exists()) {
            ret.put("found", true);
            ret.put("path", resolved.getAbsolutePath());
            ret.put("name", resolved.getName());
            ret.put("size", resolved.length());
        } else {
            ret.put("found", false);
            ret.put("path", inputPath);
        }
        call.resolve(ret);
    }

    @PluginMethod
    public void scanForModels(PluginCall call) {
        executor.execute(() -> {
            JSArray models = new JSArray();
            try {
                boolean hasPermission = true;
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                    hasPermission = Environment.isExternalStorageManager();
                }

                if (!hasPermission) {
                    JSObject ret = new JSObject();
                    ret.put("models", models);
                    ret.put("permissionRequired", true);
                    call.resolve(ret);
                    return;
                }

                File ext = Environment.getExternalStorageDirectory();
                File dl = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS);
                File docs = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS);

                List<File> dirsToScan = new ArrayList<>();
                if (dl != null && dl.exists()) dirsToScan.add(dl);
                if (docs != null && docs.exists()) dirsToScan.add(docs);
                if (ext != null && ext.exists()) {
                    File dls = new File(ext, "Downloads");
                    if (dls.exists()) dirsToScan.add(dls);
                    File tg1 = new File(ext, "Download/Telegram");
                    if (tg1.exists()) dirsToScan.add(tg1);
                    File tg2 = new File(ext, "Telegram/Telegram Documents");
                    if (tg2.exists()) dirsToScan.add(tg2);
                    File m = new File(ext, "Models");
                    if (m.exists()) dirsToScan.add(m);
                    File llm = new File(ext, "LLM");
                    if (llm.exists()) dirsToScan.add(llm);
                    dirsToScan.add(ext);
                }

                Set<String> seenPaths = new HashSet<>();
                List<File> found = new ArrayList<>();

                for (File dir : dirsToScan) {
                    File[] files = dir.listFiles();
                    if (files != null) {
                        for (File f : files) {
                            if (f.isFile() && (f.getName().toLowerCase().endsWith(".gguf") || f.getName().toLowerCase().endsWith(".bin"))) {
                                if (seenPaths.add(f.getAbsolutePath())) {
                                    found.add(f);
                                }
                            }
                        }
                    }
                }

                for (File f : found) {
                    JSObject item = new JSObject();
                    item.put("name", f.getName());
                    item.put("path", f.getAbsolutePath());
                    item.put("size", f.length());
                    long bytes = f.length();
                    double gb = bytes / (1024.0 * 1024.0 * 1024.0);
                    String sizeFormatted = gb >= 1.0
                            ? String.format(Locale.US, "%.2f ГБ", gb)
                            : String.format(Locale.US, "%.1f МБ", bytes / (1024.0 * 1024.0));
                    item.put("sizeFormatted", sizeFormatted);
                    models.put(item);
                }

                JSObject ret = new JSObject();
                ret.put("models", models);
                ret.put("permissionRequired", false);
                call.resolve(ret);
            } catch (Exception e) {
                Log.e(TAG, "Ошибка сканирования моделей: " + e.getMessage());
                call.reject("Ошибка сканирования моделей: " + e.getMessage());
            }
        });
    }

    @PluginMethod
    public void loadModel(PluginCall call) {
        String path = call.getString("path");
        if (path == null || path.isEmpty()) {
            call.reject("Путь к файлу модели GGUF не указан");
            return;
        }

        File file = resolveModelFile(path);
        if (file == null || !file.exists()) {
            boolean hasPermission = true;
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                hasPermission = Environment.isExternalStorageManager();
            }

            if (!hasPermission) {
                call.reject("Приложению не предоставлен доступ к файлам телефона. Откройте Настройки -> Приложения -> Argent Mobile -> Разрешения -> Доступ ко всем файлам и включите его.");
            } else {
                call.reject("Файл модели '" + path + "' не найден на телефоне. Убедитесь, что файл скачан и помещён в папку 'Загрузки' (Download), Документы или Telegram.");
            }
            return;
        }

        final String finalPath = file.getAbsolutePath();
        final int threads = call.getInt("threads", 4);
        final int context = call.getInt("context", 2048);

        Log.i(TAG, "Загрузка модели из: " + finalPath + " (размер: " + file.length() + " байт)");

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
                    call.reject("Не удалось инициализировать модель GGUF в нативном движке llama.cpp (возможно, файл повреждён или не поддерживается)");
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

                long genStart = System.currentTimeMillis();
                int tokensCount = nativeGenerate(prompt, temperature, maxTokens, callback);
                long genElapsed = System.currentTimeMillis() - genStart;
                double tps = (tokensCount > 0 && genElapsed > 0) ? (tokensCount * 1000.0) / genElapsed : 0.0;
                Log.i(TAG, String.format(Locale.US, "Генерация завершена: токенов=%d, время=%d мс (%.2f ток/с)", tokensCount, genElapsed, tps));

                if (tokensCount < 0) {
                    call.reject("Ошибка выполнения генерации в движке llama.cpp (декодирование прервано)");
                    return;
                }

                JSObject ret = new JSObject();
                ret.put("success", true);
                ret.put("tokensGenerated", tokensCount);
                ret.put("durationMs", genElapsed);
                ret.put("tokensPerSecond", tps);
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
