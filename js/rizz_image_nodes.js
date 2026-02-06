import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const HIDDEN_TAG = "tschide";

app.registerExtension({
    name: "rizznodes.image_nodes",

    nodeCreated(node) {
        if (node.comfyClass === "RizzSaveImage" || node.comfyClass === "RizzPreviewImage" || node.comfyClass === "RizzLoadImage") {
            setupImageNode(node);
        }

        // Special handling for RizzPreviewImage to ensure it's visible
        if (node.comfyClass === "RizzPreviewImage") {
            // Ensure the node has minimum size to show the preview
            if (node.size[0] < 200 || node.size[1] < 200) {
                node.setSize([256, 256]); // Set a reasonable default size for preview
            }
        }
    }
});

function setupImageNode(node) {
    if (node.__rizz_image_setup) return;
    if (!node.widgets) {
        setTimeout(() => setupImageNode(node), 50);
        return;
    }
    node.__rizz_image_setup = true;

    const cacheKey = `Comfy.RizzNodes.ImageCache.${node.id}`;

    // Auto-resize node to image on execution (for Save/Preview nodes)
    const onExecuted = node.onExecuted;
    node.onExecuted = function (message) {
        if (onExecuted) onExecuted.apply(this, arguments);

        let changed = false;
        const uiImages = message?.images || message?.ui?.images || this.ui?.images;
        if (uiImages && uiImages.length > 0) {
            if (!this.properties) this.properties = {};
            this.properties.last_output = uiImages;
            changed = true;
            try {
                localStorage.setItem(cacheKey, JSON.stringify(uiImages));
            } catch (e) { }
        }

        if (this.imgs && this.imgs.length > 0) {
            const img = this.imgs[0];
            // Resize node to match image (plus header space)
            const HEADER_HEIGHT = 40; // rough estimate

            if (img.width && img.height) {
                const WIDGETS_HEIGHT = this.comfyClass === "RizzLoadImage" ? 260 : 60; // Increased space for widgets to accommodate refresh button and file selector
                const OFFSET = 30; // User requested 30px further down
                const newSize = [img.width, img.height + HEADER_HEIGHT + WIDGETS_HEIGHT + OFFSET];
                this.setSize(newSize);
                this.setDirtyCanvas(true, true);

                // Save state
                if (!this.properties) this.properties = {};
                this.properties.last_size = newSize;
                changed = true;
            }

        }

        if (changed) {
            // Determine if we need to trigger an autosave
            // ComfyUI autosaves on graph change.
            // We can try to trigger it.
            // app.graph.change() might be internal.
            // But generally modifying properties doesn't always trigger autosave instantly.
            // Let's force it if possible, or just hope the user does something else?
            // The user specifically complained it doesn't save on reload.
            // So we MUST trigger persistence.

            try {
                if (this.graph?.change) this.graph.change();
            } catch (e) { }
        }
    };

    // Restore state on load
    const loadFromImages = (images) => {
        if (!images || images.length === 0) return false;
        const img_data = images[0];
        const img = new Image();
        img.onload = function () {
            node.imgs = [img];
            app.graph.setDirtyCanvas(true);
        };
        img.onerror = function () {
            if (node.properties) {
                delete node.properties.last_output;
            }
            try {
                localStorage.removeItem(cacheKey);
            } catch (e) { }
            node.imgs = null;
            app.graph.setDirtyCanvas(true);
        };
        let params = new URLSearchParams({
            filename: img_data.filename,
            type: img_data.type,
            subfolder: img_data.subfolder,
        });
        img.src = api.apiURL("/view?" + params.toString());
        return true;
    };

    if (node.properties && node.properties.last_output) {
        loadFromImages(node.properties.last_output);
    } else {
        try {
            const cached = localStorage.getItem(cacheKey);
            if (cached) {
                const parsed = JSON.parse(cached);
                loadFromImages(parsed);
            }
        } catch (e) { }
    }

    if (node.properties && node.properties.last_size) {
        node.setSize(node.properties.last_size);
    }

    // Cache widgets for toggling
    node.cachedWidgets = {};
    const widthWidget = node.widgets.find(w => w.name === "width");
    const heightWidget = node.widgets.find(w => w.name === "height");
    const resizeWidget = node.widgets.find(w => w.name === "resize");

    if (!widthWidget || !heightWidget || !resizeWidget) {
        return; // Something wrong, maybe inputs changed
    }

    // Preserve original properties if not already saved
    if (!widthWidget.origType) widthWidget.origType = widthWidget.type;
    if (!widthWidget.origComputeSize) widthWidget.origComputeSize = widthWidget.computeSize;
    if (!heightWidget.origType) heightWidget.origType = heightWidget.type;
    if (!heightWidget.origComputeSize) heightWidget.origComputeSize = heightWidget.computeSize;

    function updateVisibility() {
        // Ensure values are correct
        const isResize = resizeWidget.value === true;

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

        setVisible(widthWidget, isResize);
        setVisible(heightWidget, isResize);

        // Only auto-resize if we don't have a saved size or an image displayed
        // If we have an image (this.imgs), preserve current size or computed size from image
        // If we just loaded (last_size exists), preserve it

        // However, toggling widgets changes the needed height. 
        // If we have a massive image, and we toggle widgets, we might want to keep the massive, or adjust slightly?
        // ComfyUI default: setSize(computeSize()) shrinks to minimum.

        const currentSize = node.size;
        const minSize = node.computeSize();

        // If current size is significantly larger than minSize, it's probably sizing an image.
        // Don't shrink it.
        // But if we hide widgets, we might want to shrink slightly? No, usually fine to stay big.

        if (currentSize[0] < minSize[0] || currentSize[1] < minSize[1]) {
            node.setSize(minSize);
        }

        // Force redraw
        requestAnimationFrame(() => {
            node.setDirtyCanvas(true, true);
        });
    }

    // Hook into resize callback
    const origCallback = resizeWidget.callback;
    resizeWidget.callback = function (v) {
        updateVisibility();
        if (origCallback) origCallback.apply(this, arguments);
    };

    // Initial state check
    setTimeout(() => {
        updateVisibility();
    }, 50);

    // RizzLoadImage specific logic is handled in rizznodes_loader.js
}
