package com.argent.mobile.autopilot;

import android.accessibilityservice.AccessibilityService;
import android.accessibilityservice.GestureDescription;
import android.graphics.Path;
import android.graphics.Rect;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.accessibility.AccessibilityEvent;
import android.view.accessibility.AccessibilityNodeInfo;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;

/**
 * Системная служба специальных возможностей для автопилота Argent Mobile.
 * Обеспечивает чтение элементов интерфейса экрана и выполнение автоматических действий.
 */
public class ArgentAccessibilityService extends AccessibilityService {
    private static final String TAG = "ArgentAccessibility";
    private static ArgentAccessibilityService instance = null;

    public static boolean isRunning() {
        return instance != null;
    }

    public static ArgentAccessibilityService getInstance() {
        return instance;
    }

    @Override
    protected void onServiceConnected() {
        super.onServiceConnected();
        instance = this;
        Log.i(TAG, "ArgentAccessibilityService успешно подключена и активна");
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (instance == this) {
            instance = null;
        }
        Log.i(TAG, "ArgentAccessibilityService остановлена");
    }

    @Override
    public void onAccessibilityEvent(AccessibilityEvent event) {
        // События изменения окон и фокуса
    }

    @Override
    public void onInterrupt() {
        Log.w(TAG, "ArgentAccessibilityService прервана");
    }

    /**
     * Считывает текущее дерево элементов экрана и возвращает их в виде JSON массива.
     */
    public JSONArray dumpScreenElements() {
        JSONArray elements = new JSONArray();
        AccessibilityNodeInfo root = getRootInActiveWindow();
        if (root == null) {
            return elements;
        }

        try {
            traverseNode(root, elements);
        } finally {
            root.recycle();
        }
        return elements;
    }

    private void traverseNode(AccessibilityNodeInfo node, JSONArray array) {
        if (node == null) return;

        try {
            CharSequence text = node.getText();
            CharSequence desc = node.getContentDescription();
            String viewId = node.getViewIdResourceName();
            boolean isClickable = node.isClickable();
            boolean isEditable = node.isEditable();

            // Сохраняем информативные элементы (с текстом, описанием или кликабельные)
            if ((text != null && text.length() > 0) || 
                (desc != null && desc.length() > 0) || 
                isClickable || isEditable) {
                
                Rect bounds = new Rect();
                node.getBoundsInScreen(bounds);

                if (bounds.width() > 0 && bounds.height() > 0) {
                    JSONObject item = new JSONObject();
                    item.put("text", text != null ? text.toString() : "");
                    item.put("desc", desc != null ? desc.toString() : "");
                    item.put("id", viewId != null ? viewId : "");
                    item.put("clickable", isClickable);
                    item.put("editable", isEditable);
                    item.put("x", bounds.centerX());
                    item.put("y", bounds.centerY());
                    item.put("left", bounds.left);
                    item.put("top", bounds.top);
                    item.put("right", bounds.right);
                    item.put("bottom", bounds.bottom);
                    array.put(item);
                }
            }

            for (int i = 0; i < node.getChildCount(); i++) {
                AccessibilityNodeInfo child = node.getChild(i);
                if (child != null) {
                    traverseNode(child, array);
                    child.recycle();
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "Ошибка обхода дерева узлов: " + e.getMessage());
        }
    }

    /**
     * Виртуальное нажатие (клик) по координатам X, Y на экране.
     */
    public boolean clickAt(float x, float y) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.N) return false;

        Path clickPath = new Path();
        clickPath.moveTo(x, y);

        GestureDescription.StrokeDescription clickStroke =
                new GestureDescription.StrokeDescription(clickPath, 0, 50);
        GestureDescription.Builder gestureBuilder = new GestureDescription.Builder();
        gestureBuilder.addStroke(clickStroke);

        return dispatchGesture(gestureBuilder.build(), null, null);
    }

    /**
     * Поиск элемента по тексту или описанию и нажатие на него.
     */
    public boolean clickByText(String targetText) {
        AccessibilityNodeInfo root = getRootInActiveWindow();
        if (root == null) return false;

        try {
            List<AccessibilityNodeInfo> nodes = root.findAccessibilityNodeInfosByText(targetText);
            if (nodes != null && !nodes.isEmpty()) {
                for (AccessibilityNodeInfo node : nodes) {
                    if (performClickOnNode(node)) {
                        return true;
                    }
                }
            }
        } finally {
            root.recycle();
        }
        return false;
    }

    private boolean performClickOnNode(AccessibilityNodeInfo node) {
        if (node == null) return false;
        if (node.isClickable()) {
            return node.performAction(AccessibilityNodeInfo.ACTION_CLICK);
        }
        AccessibilityNodeInfo parent = node.getParent();
        if (parent != null) {
            boolean res = performClickOnNode(parent);
            parent.recycle();
            return res;
        }
        return false;
    }

    /**
     * Ввод текста в активное поле ввода.
     */
    public boolean inputText(String text) {
        AccessibilityNodeInfo root = getRootInActiveWindow();
        if (root == null) return false;

        try {
            AccessibilityNodeInfo focused = root.findFocus(AccessibilityNodeInfo.FOCUS_INPUT);
            if (focused != null) {
                Bundle args = new Bundle();
                args.putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text);
                boolean res = focused.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args);
                focused.recycle();
                return res;
            }
        } finally {
            root.recycle();
        }
        return false;
    }

    /**
     * Виртуальный свайп / скролл (например, вверх или вниз).
     */
    public boolean swipe(float startX, float startY, float endX, float endY, long duration) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.N) return false;

        Path swipePath = new Path();
        swipePath.moveTo(startX, startY);
        swipePath.lineTo(endX, endY);

        GestureDescription.StrokeDescription stroke =
                new GestureDescription.StrokeDescription(swipePath, 0, duration);
        GestureDescription.Builder builder = new GestureDescription.Builder();
        builder.addStroke(stroke);

        return dispatchGesture(builder.build(), null, null);
    }
}
