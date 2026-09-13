package com.argent.mobile.autopilot;

import android.accessibilityservice.AccessibilityService;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.graphics.Point;
import android.os.Build;
import android.provider.Settings;
import android.view.Display;
import android.view.WindowManager;

import com.getcapacitor.JSArray;
import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;

import org.json.JSONArray;

import java.util.List;

@CapacitorPlugin(name = "ArgentAutopilot")
public class ArgentAutopilotPlugin extends Plugin {

    @PluginMethod
    public void checkAccessibilityStatus(PluginCall call) {
        JSObject ret = new JSObject();
        boolean running = ArgentAccessibilityService.isRunning();
        ret.put("enabled", running);
        call.resolve(ret);
    }

    @PluginMethod
    public void openAccessibilitySettings(PluginCall call) {
        Intent intent = new Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS);
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        getContext().startActivity(intent);
        
        JSObject ret = new JSObject();
        ret.put("opened", true);
        call.resolve(ret);
    }

    @PluginMethod
    public void getScreenElements(PluginCall call) {
        ArgentAccessibilityService service = ArgentAccessibilityService.getInstance();
        if (service == null) {
            call.reject("Служба доступности Argent не включена в настройках Android");
            return;
        }

        try {
            JSONArray elements = service.dumpScreenElements();
            JSObject ret = new JSObject();
            ret.put("elements", elements);
            call.resolve(ret);
        } catch (Exception e) {
            call.reject("Ошибка получения дерева экрана: " + e.getMessage());
        }
    }

    @PluginMethod
    public void clickCoordinates(PluginCall call) {
        ArgentAccessibilityService service = ArgentAccessibilityService.getInstance();
        if (service == null) {
            call.reject("Служба доступности Argent не включена");
            return;
        }

        Double x = call.getDouble("x");
        Double y = call.getDouble("y");
        if (x == null || y == null) {
            call.reject("Не указаны координаты x и y");
            return;
        }

        boolean success = service.clickAt(x.floatValue(), y.floatValue());
        JSObject ret = new JSObject();
        ret.put("success", success);
        call.resolve(ret);
    }

    @PluginMethod
    public void clickText(PluginCall call) {
        ArgentAccessibilityService service = ArgentAccessibilityService.getInstance();
        if (service == null) {
            call.reject("Служба доступности Argent не включена");
            return;
        }

        String text = call.getString("text");
        if (text == null || text.isEmpty()) {
            call.reject("Текст для клика не указан");
            return;
        }

        boolean success = service.clickByText(text);
        JSObject ret = new JSObject();
        ret.put("success", success);
        call.resolve(ret);
    }

    @PluginMethod
    public void typeText(PluginCall call) {
        ArgentAccessibilityService service = ArgentAccessibilityService.getInstance();
        if (service == null) {
            call.reject("Служба доступности Argent не включена");
            return;
        }

        String text = call.getString("text", "");
        boolean success = service.inputText(text);
        JSObject ret = new JSObject();
        ret.put("success", success);
        call.resolve(ret);
    }

    @PluginMethod
    public void scroll(PluginCall call) {
        ArgentAccessibilityService service = ArgentAccessibilityService.getInstance();
        if (service == null) {
            call.reject("Служба доступности Argent не включена");
            return;
        }

        String direction = call.getString("direction", "down");
        
        WindowManager wm = (WindowManager) getContext().getSystemService(Context.WINDOW_SERVICE);
        Point size = new Point();
        if (wm != null) {
            Display display = wm.getDefaultDisplay();
            display.getSize(size);
        } else {
            size.set(1080, 2400);
        }

        float midX = size.x / 2.0f;
        float startY = size.y * 0.7f;
        float endY = size.y * 0.3f;

        if ("up".equalsIgnoreCase(direction)) {
            startY = size.y * 0.3f;
            endY = size.y * 0.7f;
        }

        boolean success = service.swipe(midX, startY, midX, endY, 300);
        JSObject ret = new JSObject();
        ret.put("success", success);
        call.resolve(ret);
    }

    @PluginMethod
    public void launchApp(PluginCall call) {
        String appNameOrPkg = call.getString("app");
        if (appNameOrPkg == null || appNameOrPkg.isEmpty()) {
            call.reject("Имя приложения не указано");
            return;
        }

        PackageManager pm = getContext().getPackageManager();
        Intent intent = pm.getLaunchIntentForPackage(appNameOrPkg);

        // Если передали не package id, а понятное имя (например "YouTube" или "Telegram")
        if (intent == null) {
            List<ApplicationInfo> packages = pm.getInstalledApplications(PackageManager.GET_META_DATA);
            for (ApplicationInfo packageInfo : packages) {
                String label = pm.getApplicationLabel(packageInfo).toString();
                if (label.equalsIgnoreCase(appNameOrPkg) || label.toLowerCase().contains(appNameOrPkg.toLowerCase())) {
                    intent = pm.getLaunchIntentForPackage(packageInfo.packageName);
                    break;
                }
            }
        }

        if (intent != null) {
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            getContext().startActivity(intent);
            JSObject ret = new JSObject();
            ret.put("launched", true);
            call.resolve(ret);
        } else {
            call.reject("Приложение не найдено на устройстве: " + appNameOrPkg);
        }
    }

    @PluginMethod
    public void pressHome(PluginCall call) {
        ArgentAccessibilityService service = ArgentAccessibilityService.getInstance();
        boolean res = false;
        if (service != null) {
            res = service.performGlobalAction(AccessibilityService.GLOBAL_ACTION_HOME);
        }
        JSObject ret = new JSObject();
        ret.put("success", res);
        call.resolve(ret);
    }

    @PluginMethod
    public void pressBack(PluginCall call) {
        ArgentAccessibilityService service = ArgentAccessibilityService.getInstance();
        boolean res = false;
        if (service != null) {
            res = service.performGlobalAction(AccessibilityService.GLOBAL_ACTION_BACK);
        }
        JSObject ret = new JSObject();
        ret.put("success", res);
        call.resolve(ret);
    }
}
