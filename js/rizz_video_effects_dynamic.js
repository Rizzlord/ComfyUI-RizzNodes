import { app } from "../../scripts/app.js";

console.log("★★★ rizz_video_effects_dynamic.js: Dynamic Widgets Loaded ★★★");

const HIDDEN_TAG = "tschide";

app.registerExtension({
    name: "rizznodes.video_effects_dynamic",

    nodeCreated(node) {
        if (node.comfyClass === "RizzVideoEffects") {
            setupVideoEffects(node);
        } else if (node.comfyClass === "RizzEditClips") {
            setupEditClips(node);
        }
    }
});

function setupEditClips(node) {
    if (!node.properties) node.properties = {};
    if (node.properties["visibleVideoCount"] === undefined) node.properties["visibleVideoCount"] = 1;

    node.cachedWidgets = {
        fixed: [],
        transitions: {}
    };

    // Cache for input slots
    node.cachedInputs = {
        video: {},
        fixed: []
    };

    let cacheReady = false;

    const hideCountWidget = () => {
        const countWidget = node.widgets?.find(w => w.name === "video_count");
        if (countWidget) {
            if (!countWidget.origType) {
                countWidget.origType = countWidget.type;
                countWidget.origComputeSize = countWidget.computeSize;
            }
            countWidget.type = HIDDEN_TAG;
            countWidget.computeSize = () => [0, -4];
            node.cachedVideoCount = countWidget;
        }
    };

    const initCache = () => {
        if (cacheReady) return;
        const allWidgets = [...node.widgets];
        const allInputs = node.inputs ? [...node.inputs] : [];

        hideCountWidget();

        // Cache fixed widgets
        const fixedNames = ["trim_start", "trim_end", "process_clips", "zoom_in"];
        for (const name of fixedNames) {
            const w = allWidgets.find(w => w.name === name);
            if (w) node.cachedWidgets.fixed.push(w);
        }

        // Cache dynamic video inputs and transition widgets
        // RizzEditClips defines video_1 ... video_25
        // Transitions start at 2
        for (let i = 1; i <= 25; i++) {
            const vidInput = allInputs.find(inp => inp.name === `video_${i}`);
            if (vidInput) {
                node.cachedInputs.video[i] = vidInput;
            }

            if (i > 1) {
                const wTrans = allWidgets.find(w => w.name === `transition_${i}`);
                const wLen = allWidgets.find(w => w.name === `trans_len_${i}`);
                if (wTrans && wLen) {
                    node.cachedWidgets.transitions[i] = { trans: wTrans, len: wLen };
                }
            }
        }

        cacheReady = true;
    };

    const ensureVideoCountWidget = () => {
        const name = "🎥 Video Clips";
        let w = node.widgets.find(x => x.name === name);
        if (!w) {
            // Generate values 1-25
            const values = Array.from({ length: 25 }, (_, i) => (i + 1).toString());
            w = node.addWidget("combo", name, "1", (v) => {
                const num = parseInt(v);
                if (!isNaN(num)) {
                    node.properties["visibleVideoCount"] = num;
                    if (node.cachedVideoCount) node.cachedVideoCount.value = num;
                    node.updateDynamicSlots();
                }
            }, { values });
        }
        w.value = node.properties["visibleVideoCount"].toString();
        if (node.cachedVideoCount) node.cachedVideoCount.value = node.properties["visibleVideoCount"];
        return w;
    };

    node.updateDynamicSlots = function () {
        if (!cacheReady) initCache();

        const videoCount = parseInt(this.properties["visibleVideoCount"] ?? 1);

        // === Update Widgets ===
        const countControl = ensureVideoCountWidget();
        this.widgets = [countControl];

        // Add hidden backend count widget
        if (node.cachedVideoCount) {
            node.cachedVideoCount.type = HIDDEN_TAG;
            node.cachedVideoCount.computeSize = () => [0, -4];
            node.cachedVideoCount.value = videoCount;
            this.widgets.push(node.cachedVideoCount);
        }

        // Add fixed widgets (trims)
        for (const w of this.cachedWidgets.fixed) {
            this.widgets.push(w);
        }

        // Add Transition Widgets for clips > 1
        for (let i = 2; i <= videoCount; i++) {
            const t = this.cachedWidgets.transitions[i];
            if (t) {
                this.widgets.push(t.trans);
                this.widgets.push(t.len);
            }
        }

        // === Update Input Connections ===
        this.inputs = [];

        // Add video inputs based on count
        for (let i = 1; i <= videoCount; i++) {
            const inp = this.cachedInputs.video[i];
            if (inp) this.inputs.push(inp);
        }

        // Resize
        const HEADER_H = 26;
        const INPUT_H = 20;
        const WIDGET_H = 21;
        const PADDING = 10;

        const inputCount = videoCount;
        const transitionWidgetCount = (videoCount > 1) ? (videoCount - 1) * 2 : 0;
        // Count + Fixed(3) + Transitions
        const widgetCount = 1 + this.cachedWidgets.fixed.length + transitionWidgetCount;

        const targetH = HEADER_H + (inputCount * INPUT_H) + (widgetCount * WIDGET_H) + PADDING;
        this.setSize([this.size[0], Math.max(100, targetH)]);

        if (app.canvas) app.canvas.setDirty(true, true);
    };

    node.onPropertyChanged = function (property, value) {
        if (property === "visibleVideoCount") {
            this.updateDynamicSlots();
        }
    };

    const origOnConfigure = node.onConfigure;
    node.onConfigure = function () {
        if (origOnConfigure) origOnConfigure.apply(this, arguments);
        setTimeout(() => node.updateDynamicSlots(), 100);
    };

    setTimeout(() => {
        initCache();
        node.updateDynamicSlots();
    }, 100);
}

function setupVideoEffects(node) {
    if (!node.properties) node.properties = {};
    if (node.properties["visibleAudioCount"] === undefined) node.properties["visibleAudioCount"] = 0;
    if (node.properties["visibleImageCount"] === undefined) node.properties["visibleImageCount"] = 0; // Default to 0 overlays

    node.cachedWidgets = {
        audio: {},
        image: {},
        fixed: []
    };

    // Cache for input slots
    node.cachedInputs = {
        audio: {},
        image: {},
        fixed: []
    };

    let cacheReady = false;

    const hideCountWidgets = () => {
        const audioCountWidget = node.widgets?.find(w => w.name === "audio_count");
        if (audioCountWidget) {
            if (!audioCountWidget.origType) {
                audioCountWidget.origType = audioCountWidget.type;
                audioCountWidget.origComputeSize = audioCountWidget.computeSize;
            }
            audioCountWidget.type = HIDDEN_TAG;
            audioCountWidget.computeSize = () => [0, -4];
            node.cachedAudioCount = audioCountWidget;
        }

        const imageCountWidget = node.widgets?.find(w => w.name === "image_count");
        if (imageCountWidget) {
            if (!imageCountWidget.origType) {
                imageCountWidget.origType = imageCountWidget.type;
                imageCountWidget.origComputeSize = imageCountWidget.computeSize;
            }
            imageCountWidget.type = HIDDEN_TAG;
            imageCountWidget.computeSize = () => [0, -4];
            node.cachedImageCount = imageCountWidget;
        }
    };

    const initCache = () => {
        if (cacheReady) return;
        const allWidgets = [...node.widgets];
        const allInputs = node.inputs ? [...node.inputs] : [];

        hideCountWidgets();

        // Cache fixed widgets
        const fixedNames = ["speed", "interpolation_mode", "reverse", "fade_in", "fade_out", "brightness", "contrast", "saturation", "end_with_audio"];
        for (const name of fixedNames) {
            const w = allWidgets.find(w => w.name === name);
            if (w) node.cachedWidgets.fixed.push(w);
        }

        // Cache audio widgets and inputs
        for (let i = 1; i <= 10; i++) {
            const wStart = allWidgets.find(w => w.name === `audio_${i}_start`);
            const wVolume = allWidgets.find(w => w.name === `audio_${i}_volume`);
            if (wStart && wVolume) {
                node.cachedWidgets.audio[i] = { start: wStart, volume: wVolume };
            }
            // Cache audio input slot
            const audioInput = allInputs.find(inp => inp.name === `audio_${i}`);
            if (audioInput) {
                node.cachedInputs.audio[i] = audioInput;
            }
        }

        // Cache image widgets and inputs
        for (let i = 1; i <= 5; i++) {
            const wBlend = allWidgets.find(w => w.name === `image_${i}_blend`);
            const wOpacity = allWidgets.find(w => w.name === `image_${i}_opacity`);
            const wPosition = allWidgets.find(w => w.name === `image_${i}_position`);
            const wTileScale = allWidgets.find(w => w.name === `image_${i}_tile_scale`);
            if (wBlend && wOpacity && wPosition) {
                node.cachedWidgets.image[i] = { blend: wBlend, opacity: wOpacity, position: wPosition, tileScale: wTileScale };
            }
            // Cache image input slot
            const imageInput = allInputs.find(inp => inp.name === `image_${i}`);
            if (imageInput) {
                node.cachedInputs.image[i] = imageInput;
            }
        }

        // Cache the video input (always visible)
        const videoInput = allInputs.find(inp => inp.name === "video");
        if (videoInput) {
            node.cachedInputs.fixed.push(videoInput);
        }

        cacheReady = true;
    };

    const ensureAudioCountWidget = () => {
        const name = "🎵 Audio Tracks";
        let w = node.widgets.find(x => x.name === name);
        if (!w) {
            const values = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"];
            w = node.addWidget("combo", name, "0", (v) => {
                const num = parseInt(v);
                if (!isNaN(num)) {
                    node.properties["visibleAudioCount"] = num;
                    if (node.cachedAudioCount) node.cachedAudioCount.value = num;
                    node.updateDynamicSlots();
                }
            }, { values });
        }
        w.value = node.properties["visibleAudioCount"].toString();
        if (node.cachedAudioCount) node.cachedAudioCount.value = node.properties["visibleAudioCount"];
        return w;
    };

    const ensureImageCountWidget = () => {
        const name = "🖼️ Image Overlays";
        let w = node.widgets.find(x => x.name === name);
        if (!w) {
            const values = ["0", "1", "2", "3", "4", "5"];
            w = node.addWidget("combo", name, "0", (v) => {
                const num = parseInt(v);
                if (!isNaN(num)) {
                    node.properties["visibleImageCount"] = num;
                    if (node.cachedImageCount) node.cachedImageCount.value = num;
                    node.updateDynamicSlots();
                }
            }, { values });
        }
        w.value = node.properties["visibleImageCount"].toString();
        if (node.cachedImageCount) node.cachedImageCount.value = node.properties["visibleImageCount"];
        return w;
    };

    node.updateDynamicSlots = function () {
        if (!cacheReady) initCache();

        const audioCount = parseInt(this.properties["visibleAudioCount"] ?? 1);
        const imageCount = parseInt(this.properties["visibleImageCount"] ?? 0);

        // === Update Widgets ===
        const audioCountControl = ensureAudioCountWidget();
        const imageCountControl = ensureImageCountWidget();

        this.widgets = [audioCountControl, imageCountControl];

        // Add hidden backend count widgets
        if (node.cachedAudioCount) {
            node.cachedAudioCount.type = HIDDEN_TAG;
            node.cachedAudioCount.computeSize = () => [0, -4];
            node.cachedAudioCount.value = audioCount;
            this.widgets.push(node.cachedAudioCount);
        }
        if (node.cachedImageCount) {
            node.cachedImageCount.type = HIDDEN_TAG;
            node.cachedImageCount.computeSize = () => [0, -4];
            node.cachedImageCount.value = imageCount;
            this.widgets.push(node.cachedImageCount);
        }

        // Add audio widgets based on count
        for (let i = 1; i <= audioCount; i++) {
            const slot = this.cachedWidgets.audio[i];
            if (slot) {
                if (slot.start) this.widgets.push(slot.start);
                if (slot.volume) this.widgets.push(slot.volume);
            }
        }

        // Add image widgets based on count
        for (let i = 1; i <= imageCount; i++) {
            const slot = this.cachedWidgets.image[i];
            if (slot) {
                if (slot.blend) this.widgets.push(slot.blend);
                if (slot.opacity) this.widgets.push(slot.opacity);
                if (slot.position) this.widgets.push(slot.position);
                if (slot.tileScale) this.widgets.push(slot.tileScale);
            }
        }

        // Add fixed widgets
        for (const w of this.cachedWidgets.fixed) {
            this.widgets.push(w);
        }

        // === Update Input Connections ===
        // Rebuild inputs array with only visible inputs
        // Note: We use a temp array and then set this.inputs to avoid reference issues
        const newInputs = [];

        // Always add video input first
        for (const inp of this.cachedInputs.fixed) {
            newInputs.push(inp);
        }

        // Add audio inputs based on count
        for (let i = 1; i <= audioCount; i++) {
            const inp = this.cachedInputs.audio[i];
            if (inp) newInputs.push(inp);
        }

        // Add image inputs based on count
        for (let i = 1; i <= imageCount; i++) {
            const inp = this.cachedInputs.image[i];
            if (inp) newInputs.push(inp);
        }

        this.inputs = newInputs;

        // Calculate size - account for inputs too
        const HEADER_H = 26;
        const INPUT_H = 20;  // Height per input connection
        const WIDGET_H = 21;
        const PADDING = 10;

        const inputCount = 1 + audioCount + imageCount;  // video + audio + image inputs
        // audio: 2 widgets per track (start, volume)
        // image: 4 widgets per layer (blend, opacity, position, tile_scale)
        //固定: speed, interpolation, reverse, fade_in, fade_out, brightness, contrast, saturation, end_with_audio (9 total)
        const widgetCount = 2 + audioCount * 2 + imageCount * 4 + this.cachedWidgets.fixed.length;

        const targetH = HEADER_H + (inputCount * INPUT_H) + (widgetCount * WIDGET_H) + PADDING;
        this.setSize([this.size[0], Math.max(180, targetH)]);

        if (app.canvas) app.canvas.setDirty(true, true);
    };

    node.onPropertyChanged = function (property, value) {
        if (property === "visibleAudioCount" || property === "visibleImageCount") {
            this.updateDynamicSlots();
        }
    };

    const origOnConfigure = node.onConfigure;
    node.onConfigure = function () {
        if (origOnConfigure) origOnConfigure.apply(this, arguments);
        setTimeout(() => node.updateDynamicSlots(), 100);
    };

    setTimeout(() => {
        initCache();
        node.updateDynamicSlots();
    }, 100);
}
