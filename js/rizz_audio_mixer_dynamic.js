import { app } from "../../scripts/app.js";

console.log("★★★ rizz_audio_mixer_dynamic.js: Dynamic Widgets Loaded ★★★");

const HIDDEN_TAG = "tschide";

app.registerExtension({
    name: "rizznodes.audio_mixer_dynamic",

    nodeCreated(node) {
        if (node.comfyClass === "RizzAudioMixer") {
            setupAudioMixer(node);
        }
    }
});

function setupAudioMixer(node) {
    if (!node.properties) node.properties = {};
    if (node.properties["visibleAudioCount"] === undefined) node.properties["visibleAudioCount"] = 2; // Default to 2

    node.cachedWidgets = {
        audio: {}
    };

    // Cache for input slots
    node.cachedInputs = {
        audio: {}
    };

    node.cachedWidgets.fixed = []; // For global inputs
    node.cachedWidgets.audio1 = {}; // Special cache for audio 1 widgets

    let cacheReady = false;

    // Helper to hide the original integer widget so we can replace it or manage it
    const hideCountWidget = () => {
        const countWidget = node.widgets?.find(w => w.name === "audio_count");
        if (countWidget) {
            if (!countWidget.origType) {
                countWidget.origType = countWidget.type;
                countWidget.origComputeSize = countWidget.computeSize;
            }
            countWidget.type = HIDDEN_TAG;
            countWidget.computeSize = () => [0, -4];
            node.cachedCountWidget = countWidget;
        }
    };

    const initCache = () => {
        if (cacheReady) return;
        const allWidgets = [...node.widgets];
        const allInputs = node.inputs ? [...node.inputs] : [];

        hideCountWidget();

        // Cache fixed widgets (Overall Trim)
        const fixedNames = ["overall_trim_start", "overall_trim_end"];
        for (const name of fixedNames) {
            const w = allWidgets.find(w => w.name === name);
            if (w) node.cachedWidgets.fixed.push(w);
        }

        // Cache Audio 1 widgets (Fixed)
        // volume_1, repeat_1, trim_start_1, trim_end_1
        const wVol1 = allWidgets.find(w => w.name === "volume_1");
        const wRep1 = allWidgets.find(w => w.name === "repeat_1");
        const wTS1 = allWidgets.find(w => w.name === "trim_start_1");
        const wTE1 = allWidgets.find(w => w.name === "trim_end_1");
        if (wVol1) node.cachedWidgets.audio1.volume = wVol1;
        if (wRep1) node.cachedWidgets.audio1.repeat = wRep1;
        if (wTS1) node.cachedWidgets.audio1.trimStart = wTS1;
        if (wTE1) node.cachedWidgets.audio1.trimEnd = wTE1;

        // Audio 1 Input is now required, so it's likely handled by ComfyUI as a permanent input. 
        // We should find it to ensure we keep it in order if we rebuild inputs.
        const inp1 = allInputs.find(inp => inp.name === "audio_1");
        if (inp1) node.cachedInputs.audio1 = inp1;

        // RizzAudioMixer defines audio_2 ... audio_50
        // And widgets mode_2, start_time_2, volume_2, trim_start_2, trim_end_2 ...
        for (let i = 2; i <= 50; i++) {
            const wMode = allWidgets.find(w => w.name === `mode_${i}`);
            const wStart = allWidgets.find(w => w.name === `start_time_${i}`);
            const wVolume = allWidgets.find(w => w.name === `volume_${i}`);
            const wRepeat = allWidgets.find(w => w.name === `repeat_${i}`);
            const wTrimStart = allWidgets.find(w => w.name === `trim_start_${i}`);
            const wTrimEnd = allWidgets.find(w => w.name === `trim_end_${i}`);

            if (wMode && wVolume) {
                const entry = { mode: wMode, volume: wVolume };
                if (wStart) entry.startTime = wStart;
                if (wRepeat) entry.repeat = wRepeat;
                if (wTrimStart) entry.trimStart = wTrimStart;
                if (wTrimEnd) entry.trimEnd = wTrimEnd;
                node.cachedWidgets.audio[i] = entry;
            }

            const audioInput = allInputs.find(inp => inp.name === `audio_${i}`);
            if (audioInput) {
                node.cachedInputs.audio[i] = audioInput;
            }
        }

        cacheReady = true;
    };

    const ensureCountWidget = () => {
        const name = "🎵 Input Count";
        let w = node.widgets.find(x => x.name === name);
        if (!w) {
            // Dropdown values 1 to 50
            const values = Array.from({ length: 50 }, (_, i) => (i + 1).toString());
            w = node.addWidget("combo", name, "2", (v) => {
                const num = parseInt(v);
                if (!isNaN(num)) {
                    node.properties["visibleAudioCount"] = num;
                    if (node.cachedCountWidget) node.cachedCountWidget.value = num;
                    node.updateDynamicSlots();
                }
            }, { values });
        }
        w.value = node.properties["visibleAudioCount"].toString();
        if (node.cachedCountWidget) node.cachedCountWidget.value = node.properties["visibleAudioCount"];
        return w;
    };

    node.updateDynamicSlots = function () {
        if (!cacheReady) initCache();

        const count = parseInt(this.properties["visibleAudioCount"] ?? 2);

        // === Update Widgets ===
        const countControl = ensureCountWidget();
        this.widgets = [countControl];

        // Add hidden backend count widget
        if (node.cachedCountWidget) {
            node.cachedCountWidget.type = HIDDEN_TAG;
            node.cachedCountWidget.computeSize = () => [0, -4];
            node.cachedCountWidget.value = count;
            this.widgets.push(node.cachedCountWidget);
        }

        // Add Audio 1 Widgets (Fixed)
        if (node.cachedWidgets.audio1.volume) this.widgets.push(node.cachedWidgets.audio1.volume);
        if (node.cachedWidgets.audio1.repeat) this.widgets.push(node.cachedWidgets.audio1.repeat);
        if (node.cachedWidgets.audio1.trimStart) this.widgets.push(node.cachedWidgets.audio1.trimStart);
        if (node.cachedWidgets.audio1.trimEnd) this.widgets.push(node.cachedWidgets.audio1.trimEnd);

        // Add dynamic widgets based on count (starting from 2)
        for (let i = 2; i <= count; i++) {
            const slot = this.cachedWidgets.audio[i];
            if (slot) {
                this.widgets.push(slot.mode);
                if (slot.startTime) this.widgets.push(slot.startTime);
                this.widgets.push(slot.volume);
                if (slot.repeat) this.widgets.push(slot.repeat);
                if (slot.trimStart) this.widgets.push(slot.trimStart);
                if (slot.trimEnd) this.widgets.push(slot.trimEnd);
            }
        }

        // Add fixed/global widgets
        if (node.cachedWidgets.fixed) {
            for (const w of node.cachedWidgets.fixed) {
                this.widgets.push(w);
            }
        }

        // === Update Input Connections ===
        this.inputs = [];

        // Add Audio 1 Input (Always first)
        if (node.cachedInputs.audio1) {
            this.inputs.push(node.cachedInputs.audio1);
        }

        // Add dynamic audio inputs based on count (starting from 2)
        for (let i = 2; i <= count; i++) {
            const inp = this.cachedInputs.audio[i];
            if (inp) this.inputs.push(inp);
        }

        // Resize
        const HEADER_H = 30;
        const WIDGET_H = 26; // Reduced specific widget height closer to actual size
        const PADDING = 10; // Reduced padding

        // Count visible widgets only (exclude hidden ones)
        const visibleWidgetCount = this.widgets.filter(w => w.type !== HIDDEN_TAG).length;

        const targetH = HEADER_H + (visibleWidgetCount * WIDGET_H) + PADDING;
        this.setSize([this.size[0], Math.max(150, targetH)]);

        if (app.canvas) app.canvas.setDirty(true, true);
    };

    node.onPropertyChanged = function (property, value) {
        if (property === "visibleAudioCount") {
            this.updateDynamicSlots();
        }
    };

    // Hook into onConfigure to restore state
    const origOnConfigure = node.onConfigure;
    node.onConfigure = function () {
        if (origOnConfigure) origOnConfigure.apply(this, arguments);
        setTimeout(() => node.updateDynamicSlots(), 100);
    };

    // Initial setup
    setTimeout(() => {
        initCache();
        node.updateDynamicSlots();
    }, 100);
}
