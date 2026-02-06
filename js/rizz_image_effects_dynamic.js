import { app } from "../../scripts/app.js";

console.log("★★★ rizz_image_effects_dynamic.js: Dynamic Widgets Loaded ★★★");

const HIDDEN_TAG = "tschide";

app.registerExtension({
    name: "rizznodes.image_effects_dynamic",

    nodeCreated(node) {
        if (node.comfyClass === "RizzImageEffects") {
            setupImageEffects(node);
        }
    }
});

function setupImageEffects(node) {
    if (!node.properties) node.properties = {};
    if (node.properties["visibleImageCount"] === undefined) node.properties["visibleImageCount"] = 0; // Default to 0 overlays

    node.cachedWidgets = {
        image: {},
        fixed: []
    };

    // Cache for input slots
    node.cachedInputs = {
        image: {},
        fixed: []
    };

    let cacheReady = false;

    const hideCountWidget = () => {
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

        hideCountWidget();

        // Cache fixed widgets (none really, all are dynamic or count control, except maybe hidden ones?)
        // In RizzImageEffects, we only have dynamic widgets other than base_image input and count
        // Wait, are there any optional fixed widgets? No, RizzImageEffects has no fixed optional widgets in my implementation.
        // It only has "base_image" (required input) and "image_count" (required input).
        // And then optional dynamic ones.

        // Cache image widgets and inputs
        for (let i = 1; i <= 5; i++) {
            const wBlend = allWidgets.find(w => w.name === `image_${i}_blend`);
            const wOpacity = allWidgets.find(w => w.name === `image_${i}_opacity`);
            const wPosition = allWidgets.find(w => w.name === `image_${i}_position`);
            const wTileScale = allWidgets.find(w => w.name === `image_${i}_tile_scale`);
            if (wBlend && wOpacity && wPosition) { // Tile scale might be there or not depending on partial updates? Should be there.
                node.cachedWidgets.image[i] = {
                    blend: wBlend,
                    opacity: wOpacity,
                    position: wPosition,
                    tileScale: wTileScale
                };
            }
            // Cache image input slot
            const imageInput = allInputs.find(inp => inp.name === `image_${i}`);
            if (imageInput) {
                node.cachedInputs.image[i] = imageInput;
            }
        }

        // Cache the base_image input (always visible)
        const baseInput = allInputs.find(inp => inp.name === "base_image");
        if (baseInput) {
            node.cachedInputs.fixed.push(baseInput);
        }

        cacheReady = true;
    };

    const ensureImageCountWidget = () => {
        const name = "🖼️ Overlay Layers";
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

        const imageCount = parseInt(this.properties["visibleImageCount"] ?? 0);

        // === Update Widgets ===
        const imageCountControl = ensureImageCountWidget();

        this.widgets = [imageCountControl];

        // Add hidden backend count widget
        if (node.cachedImageCount) {
            node.cachedImageCount.type = HIDDEN_TAG;
            node.cachedImageCount.computeSize = () => [0, -4];
            node.cachedImageCount.value = imageCount;
            this.widgets.push(node.cachedImageCount);
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

        // === Update Input Connections ===
        const newInputs = [];

        // Always add base inputs
        for (const inp of this.cachedInputs.fixed) {
            newInputs.push(inp);
        }

        // Add image inputs based on count
        for (let i = 1; i <= imageCount; i++) {
            const inp = this.cachedInputs.image[i];
            if (inp) newInputs.push(inp);
        }

        this.inputs = newInputs;

        // Calculate size
        const HEADER_H = 26;
        const INPUT_H = 20;
        const WIDGET_H = 21;
        const PADDING = 10;

        const inputCount = 1 + imageCount; // base + overlays
        const widgetCount = 1 + imageCount * 4; // control + 4 per layer

        const targetH = HEADER_H + (inputCount * INPUT_H) + (widgetCount * WIDGET_H) + PADDING;
        this.setSize([this.size[0], Math.max(120, targetH)]);

        if (app.canvas) app.canvas.setDirty(true, true);
    };

    node.onPropertyChanged = function (property, value) {
        if (property === "visibleImageCount") {
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
