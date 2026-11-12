(function (global) {
  const state = {
    ready: false,
    configured: false,
    client: null,
    user: null,
    listeners: [],
  };

  function config() {
    return global.SUPABASE_CONFIG || {};
  }

  function isConfigured() {
    const cfg = config();
    return !!(cfg.url && cfg.anonKey);
  }

  function notify() {
    for (const fn of state.listeners) {
      try { fn(getSnapshot()); } catch (error) { console.error(error); }
    }
  }

  function getSnapshot() {
    return {
      configured: state.configured,
      ready: state.ready,
      user: state.user,
      email: state.user && state.user.email || "",
    };
  }

  function onAuthChange(fn) {
    state.listeners.push(fn);
    fn(getSnapshot());
    return () => {
      state.listeners = state.listeners.filter((item) => item !== fn);
    };
  }

  function authErrorMessage(error) {
    if (!error) return "Unknown error";
    const text = error.message || String(error);
    if (/not configured/i.test(text)) return "Supabase is not configured yet.";
    if (/invalid login|invalid credentials/i.test(text)) return "Wrong email or password.";
    if (/already registered|already been registered/i.test(text)) {
      return "That email is already registered — try logging in.";
    }
    if (/confirm|verification/i.test(text)) {
      return "Check your email to confirm the account, then log in.";
    }
    return text;
  }

  function sdk() {
    const lib = global.supabase;
    if (!lib || typeof lib.createClient !== "function") {
      throw new Error("Supabase SDK failed to load");
    }
    return lib;
  }

  async function init() {
    state.configured = isConfigured();
    if (!state.configured) {
      state.ready = true;
      notify();
      return getSnapshot();
    }
    const cfg = config();
    state.client = sdk().createClient(cfg.url, cfg.anonKey, {
      auth: {
        persistSession: true,
        autoRefreshToken: true,
        detectSessionInUrl: true,
      },
    });
    const { data } = await state.client.auth.getSession();
    state.user = data && data.session && data.session.user || null;
    state.client.auth.onAuthStateChange((_event, session) => {
      state.user = session && session.user || null;
      notify();
    });
    state.ready = true;
    notify();
    return getSnapshot();
  }

  async function signUp(email, password) {
    if (!state.client) throw new Error("Supabase is not configured yet.");
    const { data, error } = await state.client.auth.signUp({ email, password });
    if (error) throw error;
    state.user = data && data.user && data.session ? data.user : state.user;
    notify();
    return {
      needsEmailConfirm: !(data && data.session),
      user: state.user,
    };
  }

  async function signIn(email, password) {
    if (!state.client) throw new Error("Supabase is not configured yet.");
    const { data, error } = await state.client.auth.signInWithPassword({ email, password });
    if (error) throw error;
    state.user = data && data.user || null;
    notify();
    return state.user;
  }

  async function signOut() {
    if (!state.client) return;
    await state.client.auth.signOut();
    state.user = null;
    notify();
  }

  async function pullCloudSaves() {
    if (!state.client || !state.user) return null;
    const { data, error } = await state.client
      .from("saves")
      .select("id, city, maps_tab, updated_at")
      .order("updated_at", { ascending: false });
    if (error) throw error;
    return (data || []).map((row) => ({
      id: row.city,
      city: row.city,
      mapsTab: row.maps_tab || "all",
      t: row.updated_at ? Date.parse(row.updated_at) : Date.now(),
      _id: row.id,
    }));
  }

  async function upsertCloudSave(save) {
    if (!state.client || !state.user || !save || !save.city) return;
    const { error } = await state.client.from("saves").upsert({
      user_id: state.user.id,
      city: save.city,
      maps_tab: save.mapsTab || "all",
      updated_at: new Date(save.t || Date.now()).toISOString(),
    }, { onConflict: "user_id,city" });
    if (error) throw error;
  }

  async function deleteCloudSave(cityName) {
    if (!state.client || !state.user || !cityName) return;
    const { error } = await state.client
      .from("saves")
      .delete()
      .eq("city", cityName);
    if (error) throw error;
  }

  global.DitieAuth = {
    init,
    onAuthChange,
    getSnapshot,
    isConfigured,
    signUp,
    signIn,
    signOut,
    pullCloudSaves,
    upsertCloudSave,
    deleteCloudSave,
    authErrorMessage,
  };
})(window);
