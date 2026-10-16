package com.argent.mobile.files;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;
import android.webkit.MimeTypeMap;

import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Locale;

/**
 * Сохранение текстового файла в «Загрузки/Argent».
 *
 * В WebView приложения ссылка с атрибутом download на blob: ничего не скачивает — кнопки «Скачать» и
 * «Экспорт проекта» в APK молча не работали. Android 10+ пишет через MediaStore (разрешения не нужны),
 * старые версии — прямо в папку Download.
 */
@CapacitorPlugin(name = "ArgentFiles")
public class ArgentFilesPlugin extends Plugin {

    @PluginMethod
    public void saveText(PluginCall call) {
        String name = call.getString("name");
        String content = call.getString("content", "");
        if (name == null || name.trim().isEmpty()) {
            call.reject("Не указано имя файла");
            return;
        }
        String safe = sanitize(name);
        byte[] data = (content == null ? "" : content).getBytes(StandardCharsets.UTF_8);
        try {
            String where;
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                ContentValues values = new ContentValues();
                values.put(MediaStore.MediaColumns.DISPLAY_NAME, safe);
                values.put(MediaStore.MediaColumns.MIME_TYPE, mimeFor(safe));
                values.put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS + "/Argent");
                ContentResolver resolver = getContext().getContentResolver();
                Uri uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values);
                if (uri == null) throw new IOException("MediaStore не создал файл");
                try (OutputStream out = resolver.openOutputStream(uri)) {
                    if (out == null) throw new IOException("нет доступа на запись");
                    out.write(data);
                }
                where = Environment.DIRECTORY_DOWNLOADS + "/Argent/" + safe;
            } else {
                File dir = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "Argent");
                if (!dir.exists() && !dir.mkdirs()) throw new IOException("не удалось создать папку " + dir);
                File target = uniqueFile(dir, safe);
                try (FileOutputStream out = new FileOutputStream(target)) {
                    out.write(data);
                }
                where = target.getAbsolutePath();
            }
            JSObject ret = new JSObject();
            ret.put("path", where);
            call.resolve(ret);
        } catch (Exception e) {
            call.reject("Не удалось сохранить файл: " + e.getMessage());
        }
    }

    /** Имя без разделителей пути и запрещённых символов. */
    private static String sanitize(String name) {
        String cleaned = name.trim().replaceAll("[\\\\/:*?\"<>|\\p{Cntrl}]", "_");
        return cleaned.isEmpty() ? "argent.txt" : cleaned;
    }

    /** Тип по расширению: с text/plain MediaStore дописывает ".txt" к main.py. */
    private static String mimeFor(String name) {
        int dot = name.lastIndexOf('.');
        if (dot > 0 && dot < name.length() - 1) {
            String ext = name.substring(dot + 1).toLowerCase(Locale.ROOT);
            if (ext.equals("txt") || ext.equals("md")) return "text/plain";
            String mime = MimeTypeMap.getSingleton().getMimeTypeFromExtension(ext);
            if (mime != null) return mime;
        }
        return "application/octet-stream";
    }

    private static File uniqueFile(File dir, String name) {
        File f = new File(dir, name);
        if (!f.exists()) return f;
        int dot = name.lastIndexOf('.');
        String base = dot > 0 ? name.substring(0, dot) : name;
        String ext = dot > 0 ? name.substring(dot) : "";
        for (int i = 2; i < 1000; i++) {
            f = new File(dir, base + " (" + i + ")" + ext);
            if (!f.exists()) return f;
        }
        return new File(dir, base + "_" + System.currentTimeMillis() + ext);
    }
}
