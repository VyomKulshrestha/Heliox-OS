export async function invoke<T = any>(command: string, args?: any): Promise<T> {
  // First check if native Tauri IPC bridge is present
  if (typeof window !== "undefined" && (window as any).__TAURI_INTERNALS__) {
    try {
      const { invoke: tauriInvoke } = await import("@tauri-apps/api/core");
      return await tauriInvoke<T>(command, args);
    } catch (e) {
      console.error(`Tauri native invoke error (${command}):`, e);
      throw e;
    }
  }

  // Fallback for browser dev mode (npm run dev running in standard Chrome/Edge)
  try {
    const res = await fetch("/api/tauri_invoke", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command, args }),
    });
    if (res.ok) {
      return await res.json();
    }
  } catch (e) {
    console.error(`Dev server fallback error (${command}):`, e);
  }

  // Return safe defaults if dev server proxy returns error
  if (command === "get_system_stats") {
    return {
      cpu: 12,
      ram: 44,
      disk: 38,
      network_up: 84,
      network_down: 312,
      cpu_name: "Local CPU (Dev Mode)",
      total_ram: 16,
      disk_size: 512,
    } as unknown as T;
  }
  if (command === "get_temperature_stats") {
    return {
      cpu: 44,
      gpu: 40,
      motherboard: 36,
      ssd: 34,
      vrm: 33,
      battery: 29,
      power: 52,
      cpu_name: "Local CPU",
      cpu_threads: 16,
      battery_percent: 95
    } as unknown as T;
  }
  if (command === "get_uptime") {
    return "4h 12m" as unknown as T;
  }
  if (command === "get_log_count") {
    return 128 as unknown as T;
  }
  if (command === "get_terminal_logs") {
    return [
      "[System] Heliox OS Agent Daemon Connected",
      "[Core] ReAct Loop active on ws://127.0.0.1:8785",
      "[Monitor] System health metrics normal",
      "[Cognitive] TRIBE v2 Neural Cognitive HUD loaded",
    ] as unknown as T;
  }
  if (command === "get_agent_activity") {
    return [
      { agent: "System Agent", status: "Idle", tasks_completed: 14 },
      { agent: "Code Agent", status: "Active", tasks_completed: 8 },
      { agent: "Web Agent", status: "Idle", tasks_completed: 5 },
      { agent: "Monitor Agent", status: "Monitoring", tasks_completed: 42 },
      { agent: "Communication Agent", status: "Ready", tasks_completed: 3 },
    ] as unknown as T;
  }
  if (command === "get_rss_feed") {
    return [
      { title: "Heliox OS v0.7.1 Released with JARVIS Autonomy", url: "https://github.com/VyomKulshrestha/Heliox-OS/releases", source: "GitHub" },
      { title: "TRIBE v2 Cognitive Engine Integration Live", url: "https://helioxos.dev", source: "Heliox Blog" },
    ] as unknown as T;
  }
  if (command === "get_status_metrics") {
    return { cpu: 12, ram: 44, latency_ms: 8, agents_active: 5 } as unknown as T;
  }
  if (command === "get_dashboard_status") {
    return {
      connected: true,
      agents: 5,
      cpu: "15%",
      memory: "44%",
      network_up: "140 KB/s",
      network_down: "520 KB/s",
    } as unknown as T;
  }
  if (command === "get_auth_token") {
    try {
      const res = await fetch("/api/auth_token");
      if (res.ok) return (await res.text()).trim() as unknown as T;
    } catch {
      // ignore
    }
    return ((import.meta as any).env?.VITE_DAEMON_TOKEN ?? "") as unknown as T;
  }
  if (command === "get_hotkey") {
    return "Ctrl+Space" as unknown as T;
  }

  return {} as unknown as T;
}
