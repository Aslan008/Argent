package com.argent.mobile;

import android.os.Bundle;
import com.argent.mobile.autopilot.ArgentAutopilotPlugin;
import com.argent.mobile.llama.ArgentLlamaPlugin;
import com.getcapacitor.BridgeActivity;

public class MainActivity extends BridgeActivity {
    @Override
    public void onCreate(Bundle savedInstanceState) {
        registerPlugin(ArgentAutopilotPlugin.class);
        registerPlugin(ArgentLlamaPlugin.class);
        super.onCreate(savedInstanceState);
    }
}
