import { app } from "../../scripts/app.js";

console.log("★★★ rizz_video_effects_dynamic.js: RizzVideoEffects Dynamic Widgets ★★★");

const HIDDEN_TAG = "tschide";

app.registerExtension({
    name: "rizznodes.video_effects_dynamic",

    nodeCreated(node) {
        if (node.comfyClass !== "RizzVideoEffects") return;

        if (!node.properties) node.properties = {};
        if (node.properties["visibleAudioCount"] === undefined) node.properties["visibleAudioCount"] = 1;
        if (node.properties["visibleImageCount"] === undefined) node.properties["visibleImageCount"] = 1;

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
            const fixedNames = ["speed", "reverse", "fade_in", "fade_out", "brightness", "contrast", "saturation", "end_with_audio"];
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
                w = node.addWidget("combo", name, "1", (v) => {
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

            const audioCount = parseInt(this.properties["visibleAudioCount"] || 1);
            const imageCount = parseInt(this.properties["visibleImageCount"] || 0);

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
            this.inputs = [];

            // Always add video input first
            for (const inp of this.cachedInputs.fixed) {
                this.inputs.push(inp);
            }

            // Add audio inputs based on count
            for (let i = 1; i <= audioCount; i++) {
                const inp = this.cachedInputs.audio[i];
                if (inp) this.inputs.push(inp);
            }

            // Add image inputs based on count
            for (let i = 1; i <= imageCount; i++) {
                const inp = this.cachedInputs.image[i];
                if (inp) this.inputs.push(inp);
            }

            // Calculate size - account for inputs too
            const HEADER_H = 26;
            const INPUT_H = 22;  // Height per input connection
            const WIDGET_H = 26;
            const PADDING = 20;

            const inputCount = 1 + audioCount + imageCount;  // video + audio + image inputs
            const widgetCount = 2 + audioCount * 2 + imageCount * 3 + this.cachedWidgets.fixed.length;

            const targetH = HEADER_H + (inputCount * INPUT_H) + (widgetCount * WIDGET_H) + PADDING;
            this.setSize([this.size[0], Math.max(200, targetH)]);

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
});
