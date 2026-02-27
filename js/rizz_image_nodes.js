import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const HIDDEN_TAG = "tschide";

app.registerExtension({
    name: "rizznodes.image_nodes",

    nodeCreated(node) {
        if (node.comfyClass === "RizzSaveImage") {
            setupSaveNode(node);
        } else if (node.comfyClass === "RizzPreviewImage") {
            setupPreviewNode(node);
        } else if (node.comfyClass === "RizzLoadImage") {
            setupLoadNode(node);
        } else if (node.comfyClass === "RizzUpscaleImageBatch") {
            setupUpscaleBatchNode(node);
        } else if (node.comfyClass === "RizzEditImage") {
            setupEditImageNode(node);
        }
    }
});

// ─────────────────────────────────────────────────
// Helper: load a preview image onto the node canvas
// ─────────────────────────────────────────────────
function loadPreviewImage(node, imageInfo) {
    if (!imageInfo || !imageInfo.filename) return;
    const img = new Image();
    img.onload = function () {
        node.imgs = [img];
        node.setDirtyCanvas(true, true);
    };
    img.onerror = function () {
        // Image no longer exists – clear stale preview silently
        node.imgs = null;
        node.setDirtyCanvas(true, true);
    };
    const params = new URLSearchParams({
        filename: imageInfo.filename,
        type: imageInfo.type || "output",
        subfolder: imageInfo.subfolder || "",
    });
    img.src = api.apiURL("/view?" + params.toString());
}

// ─────────────────────────────────────────────────
// Helper: restore preview from the last execution
// ComfyUI stores execution results; we just read it
// ─────────────────────────────────────────────────
function restoreLastPreview(node) {
    // ComfyUI's built-in images array (set by onExecuted) is the canonical source.
    // If the node already has imgs from a prior execution in this session, keep them.
    if (node.imgs && node.imgs.length > 0) return;

    // Fallback: check if the node stored the last result in a lightweight property
    const lastImg = node.properties?.rizz_last_preview;
    if (lastImg) {
        loadPreviewImage(node, lastImg);
    }
}

// ─────────────────────────────────────────────────
// Helper: hook onExecuted to persist a lightweight
// preview reference (just filename/subfolder/type)
// ─────────────────────────────────────────────────
function hookOnExecuted(node, options = {}) {
    const suppressCoreImageHandling = options.suppressCoreImageHandling === true;
    const origOnExecuted = node.onExecuted;
    node.onExecuted = function (message) {
        const sizeBefore = this.size ? this.size.slice() : null;
        let messageForCore = message;

        if (suppressCoreImageHandling && message && typeof message === "object") {
            messageForCore = { ...message };
            delete messageForCore.images;
            delete messageForCore.gifs;
        }

        if (origOnExecuted) {
            if (suppressCoreImageHandling) {
                origOnExecuted.call(this, messageForCore);
            } else {
                origOnExecuted.apply(this, arguments);
            }
        }

        // Extract the image info from the execution result
        const uiImages = message?.images;
        if (uiImages && uiImages.length > 0) {
            const last = uiImages[uiImages.length - 1];
            // Store a minimal, serialization-safe reference
            if (!this.properties) this.properties = {};
            this.properties.rizz_last_preview = {
                filename: last.filename,
                subfolder: last.subfolder || "",
                type: last.type || "output",
            };

            if (suppressCoreImageHandling) {
                loadPreviewImage(this, this.properties.rizz_last_preview);
            }
        }

        // Keep manual node sizing stable when saving repeatedly.
        if (
            suppressCoreImageHandling &&
            sizeBefore &&
            this.size &&
            (this.size[0] !== sizeBefore[0] || this.size[1] !== sizeBefore[1])
        ) {
            this.setSize(sizeBefore);
        }
    };
}

// ─────────────────────────────────────────────────
// Helper: resize widget visibility toggle
// (shared by Save & Load which both have resize/width/height)
// ─────────────────────────────────────────────────
function setupResizeToggle(node) {
    const widthWidget = node.widgets?.find(w => w.name === "width");
    const heightWidget = node.widgets?.find(w => w.name === "height");
    const resizeWidget = node.widgets?.find(w => w.name === "resize");

    if (!widthWidget || !heightWidget || !resizeWidget) return;

    // Preserve original widget properties
    if (!widthWidget.origType) widthWidget.origType = widthWidget.type;
    if (!widthWidget.origComputeSize) widthWidget.origComputeSize = widthWidget.computeSize;
    if (!heightWidget.origType) heightWidget.origType = heightWidget.type;
    if (!heightWidget.origComputeSize) heightWidget.origComputeSize = heightWidget.computeSize;

    const setVisible = (w, visible) => {
        if (visible) {
            if (w.type === HIDDEN_TAG) {
                w.type = w.origType;
                w.computeSize = w.origComputeSize;
            }
        } else {
            w.type = HIDDEN_TAG;
            w.computeSize = () => [0, -4];
        }
    };

    function updateVisibility() {
        const isResize = resizeWidget.value === true;
        setVisible(widthWidget, isResize);
        setVisible(heightWidget, isResize);

        const currentSize = node.size.slice();
        const minSize = node.computeSize();

        // If the node has a custom preview image (set by rizznodes_loader.js),
        // preserve the current height so the preview isn't clipped/hidden.
        const hasCustomPreview = node.rizz_img != null;
        const newWidth = Math.max(currentSize[0], minSize[0]);
        const newHeight = hasCustomPreview
            ? Math.max(currentSize[1], minSize[1])
            : minSize[1];

        if (currentSize[0] !== newWidth || currentSize[1] !== newHeight) {
            node.setSize([newWidth, newHeight]);
        }
        requestAnimationFrame(() => node.setDirtyCanvas(true, true));
    }

    const origCallback = resizeWidget.callback;
    resizeWidget.callback = function (v) {
        updateVisibility();
        if (origCallback) origCallback.apply(this, arguments);
    };

    setTimeout(() => updateVisibility(), 50);
}

// ─────────────────────────────────────────────────
// RizzSaveImage
// ─────────────────────────────────────────────────
function setupSaveNode(node) {
    if (node.__rizz_image_setup) return;
    node.__rizz_image_setup = true;

    hookOnExecuted(node, { suppressCoreImageHandling: true });
    setupResizeToggle(node);

    // Restore preview on load / configure
    const origOnConfigure = node.onConfigure;
    node.onConfigure = function () {
        if (origOnConfigure) origOnConfigure.apply(this, arguments);
        setTimeout(() => restoreLastPreview(this), 100);
    };
    setTimeout(() => restoreLastPreview(node), 100);
}

// ─────────────────────────────────────────────────
// RizzPreviewImage
// ─────────────────────────────────────────────────
function setupPreviewNode(node) {
    if (node.__rizz_image_setup) return;
    node.__rizz_image_setup = true;

    hookOnExecuted(node);

    // Ensure minimum size for preview
    if (node.size[0] < 200 || node.size[1] < 200) {
        node.setSize([256, 256]);
    }

    // Restore preview on load / configure
    const origOnConfigure = node.onConfigure;
    node.onConfigure = function () {
        if (origOnConfigure) origOnConfigure.apply(this, arguments);
        setTimeout(() => restoreLastPreview(this), 100);
    };
    setTimeout(() => restoreLastPreview(node), 100);
}

// ─────────────────────────────────────────────────
// RizzLoadImage
// Preview is handled by rizznodes_loader.js – we
// only set up the resize/width/height toggle here.
// ─────────────────────────────────────────────────
function setupLoadNode(node) {
    if (node.__rizz_image_setup) return;
    node.__rizz_image_setup = true;
    // RizzLoadImage preview/layout is controlled in rizznodes_loader.js.
    // Keeping a second resize-toggle handler here causes size drift.
}

function setupUpscaleBatchNode(node) {
    if (node.__rizz_upscale_setup) return;
    if (!node.widgets) {
        setTimeout(() => setupUpscaleBatchNode(node), 50);
        return;
    }
    node.__rizz_upscale_setup = true;

    const fixedResolutionWidget = node.widgets.find(w => w.name === "fixed_resolution");
    if (!fixedResolutionWidget) return;

    if (!fixedResolutionWidget.origType) fixedResolutionWidget.origType = fixedResolutionWidget.type;
    if (!fixedResolutionWidget.origComputeSize) fixedResolutionWidget.origComputeSize = fixedResolutionWidget.computeSize;

    const setWidgetVisible = (visible) => {
        if (visible) {
            if (fixedResolutionWidget.type === HIDDEN_TAG) {
                fixedResolutionWidget.type = fixedResolutionWidget.origType;
                fixedResolutionWidget.computeSize = fixedResolutionWidget.origComputeSize;
            }
        } else {
            fixedResolutionWidget.type = HIDDEN_TAG;
            fixedResolutionWidget.computeSize = () => [0, -4];
        }
    };

    const hasUpscaleModelConnection = () => {
        const modelInput = node.inputs?.find(input => input.name === "upscale_model");
        if (!modelInput) return false;
        if (modelInput.link != null) return true;
        if (Array.isArray(modelInput.links) && modelInput.links.length > 0) return true;
        return false;
    };

    const updateWidgetVisibility = () => {
        const showFixedResolution = hasUpscaleModelConnection();
        setWidgetVisible(showFixedResolution);

        const minSize = node.computeSize();
        if (node.size[0] < minSize[0] || node.size[1] < minSize[1]) {
            node.setSize([Math.max(node.size[0], minSize[0]), Math.max(node.size[1], minSize[1])]);
        }
        requestAnimationFrame(() => node.setDirtyCanvas(true, true));
    };

    const origOnConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function () {
        if (origOnConnectionsChange) origOnConnectionsChange.apply(this, arguments);
        updateWidgetVisibility();
    };

    const origOnConfigure = node.onConfigure;
    node.onConfigure = function () {
        if (origOnConfigure) origOnConfigure.apply(this, arguments);
        setTimeout(updateWidgetVisibility, 0);
    };

    setTimeout(updateWidgetVisibility, 50);
}

function setupEditImageNode(node) {
    if (node.__rizz_edit_image_setup) return;
    node.__rizz_edit_image_setup = true;

    const presetWidget = node.widgets?.find(w => w.name === "color_temperature_preset");
    const originalTempWidget = node.widgets?.find(w => w.name === "original_temperature");
    const intendedTempWidget = node.widgets?.find(w => w.name === "intended_temperature");

    if (!presetWidget || !originalTempWidget || !intendedTempWidget) return;

    const presetOffsets = {
        none: 0,
        warm: 800,
        warmer: 1600,
        cold: -800,
        colder: -1600,
    };

    const clampTemperature = (value) => {
        const rounded = Math.round(Number(value) / 50) * 50;
        return Math.max(1000, Math.min(12000, rounded));
    };

    const applyPresetTemperature = () => {
        const preset = String(presetWidget.value ?? "none").toLowerCase();
        if (!(preset in presetOffsets) || preset === "none") return;

        const baseTemp = Number(originalTempWidget.value);
        if (!Number.isFinite(baseTemp)) return;

        const adjusted = clampTemperature(baseTemp + presetOffsets[preset]);
        intendedTempWidget.value = adjusted;

        if (typeof intendedTempWidget.callback === "function") {
            intendedTempWidget.callback(adjusted);
        }

        node.setDirtyCanvas(true, true);
    };

    const wrapWidgetCallback = (widget, handler) => {
        const originalCallback = widget.callback;
        widget.callback = function (v) {
            handler(v);
            if (originalCallback) originalCallback.apply(this, arguments);
        };
    };

    wrapWidgetCallback(presetWidget, applyPresetTemperature);
    wrapWidgetCallback(originalTempWidget, applyPresetTemperature);

    const origOnConfigure = node.onConfigure;
    node.onConfigure = function () {
        if (origOnConfigure) origOnConfigure.apply(this, arguments);
        setTimeout(applyPresetTemperature, 0);
    };

    setTimeout(applyPresetTemperature, 50);
}
