package com.argent.mobile;

import android.os.Bundle;
import com.argent.mobile.autopilot.ArgentAutopilotPlugin;
import com.getcapacitor.BridgeActivity;

public class MainActivity extends BridgeActivity {
    @Override
    public void onCreate(Bundle savedInstanceState) {
        registerPlugin(ArgentAutopilotPlugin.class);
        super.onCreate(savedInstanceState);
    }
}
