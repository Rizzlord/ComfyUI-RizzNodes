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

    // Auto-resize node to image on execution (for Save/Preview nodes)
    const onExecuted = node.onExecuted;
    node.onExecuted = function (message) {
        if (onExecuted) onExecuted.apply(this, arguments);

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
            }

            // Save image info for reload
            if (this.ui.images && this.ui.images.length > 0) {
                if (!this.properties) this.properties = {};
                this.properties.last_output = this.ui.images;
            }
        }
    };

    // Restore state on load
    if (node.properties && node.properties.last_output) {
        const images = node.properties.last_output;
        if (images && images.length > 0) {
            const img_data = images[0];
            const img = new Image();
            img.onload = function () {
                node.imgs = [img];
                app.graph.setDirtyCanvas(true);
            };
            // Construct URL - handle both 'output' and 'temp' types
            let params = new URLSearchParams({
                filename: img_data.filename,
                type: img_data.type,
                subfolder: img_data.subfolder,
            });
            img.src = api.apiURL("/view?" + params.toString());
        }
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

    // RizzLoadImage specific logic
    if (node.comfyClass === "RizzLoadImage") {
        // Fix duplicate image display: Disable the custom onDrawBackground that presumably renders the image
        // functionality or just rely on standard preview.
        // If the 'image' widget with image_upload=True is used, it often tries to draw the image.
        // We want to keep the one "beneath refresh files" (which is likely the ui.images return).
        // So we might need to suppress the widget's drawing or the node's background drawing.

        // Standard LoadImage uses widget.draw to draw the image? 
        // Or the node draws it. 
        // Let's try to override onDrawBackground to do nothing (or just draw standard).
        // If the duplicate is "covered by choose file", it sounds like it's drawn in the widget area.

        // Actually, "image_upload" widget DOES draw the image.
        // We can try to hide that valid image if we prefer the Review one.
        // But the user said "beneath refresh files correctly".
        // The refresh button is usually part of the widget?

        // Let's try finding the image widget and setting its 'draw' to null or similar if possible, 
        // OR (easier) simply tell the node NOT to return ui.images and rely on the widget?
        // BUT the user specifically said "only show one image... and one is beneath refresh files correctly".
        // This implies they LIKE the one at the bottom (result preview) and NOT the one in the widget area.
        // So we should suppress the widget preview.

        const imgWidget = node.widgets.find(w => w.name === "image");
        if (imgWidget) {
            // Monkey patch the widget's draw function to skip drawing the image preview?
            // The standard Comfy widget might not expose this easily, but `draw` is common.
            const origDraw = imgWidget.draw;
            imgWidget.draw = function (ctx, node, widgetWidth, y, widgetHeight) {
                // We still want the text/controls, just maybe not the big image preview?
                // Comfy's image widget is complex.
                // A safer way is to prevent it from *having* an image to draw?
                // But we need the value.

                // Alternative: The user says "covered by choose file".
                // Maybe we can just force the widget to not display the image?
                // `showImage` property?
            };
            // Actually, usually LoadImage widget draws the image if it has one.
            // If we can't easily stop it, maybe we can accept it?
            // But user says: "one is beneath ... correctly".

            // Let's try simpler approach for now:
            // The "beneath" one comes from Python `return { "ui": { "images": ... } }`.
            // The "on node" one comes from the widget.
            // If we remove the `ui` return in Python, we lose the "correct" one.
            // So we must remove the "on node" one.

            // If we set `node.images = null` maybe? No.

            // ComfyUI LoadImage widget:
            // It loads the image and stores it in `this.image` (Image object).
            // If we set `imgWidget.image = null`, it might stop drawing it?
            // But then it might reload it.
        }

        // Actually, simplest fix for "covered by choose file" is often that the node is too small?
        // But user says "show 1 image".

        // I will try to override onDrawBackground of the NODE to ensure no "extra" things are drawn 
        // if that's where it is coming from (RizzSameImage logic?).
        // RizzSaveImage logic resizing the node might be conflicting.

        // Let's implement the dynamic list first, which is the main feature.

        const folderWidget = node.widgets.find(w => w.name === "folder");
        const customPathWidget = node.widgets.find(w => w.name === "custom_path");
        // imageWidget defined above

        const refreshFiles = async () => {
            if (!folderWidget || !imgWidget) return;

            let path = folderWidget.value;
            if (path === "Custom" && customPathWidget) {
                path = customPathWidget.value;
            }

            try {
                const response = await fetch("/rizz/list_files", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ path: path, type: "image" })
                });

                if (response.ok) {
                    const data = await response.json();
                    if (data.files) {
                        imgWidget.options.values = data.files;
                        // Force update the widget to reflect the new options
                        if (data.files.length > 0) {
                            // If current value is not in list, select first
                            if (!data.files.includes(imgWidget.value)) {
                                imgWidget.value = data.files[0];
                            }
                        } else {
                            imgWidget.value = "None"; // Show "None" if no files found
                        }

                        // Trigger a redraw to make sure the dropdown is visible
                        node.setDirtyCanvas(true, true);
                    }
                }
            } catch (err) {
                console.error("[RizzNodes] Failed to fetch files:", err);
            }
        };

        if (folderWidget) {
            folderWidget.callback = () => {
                refreshFiles();
            };
        }

        if (customPathWidget) {
            // Debounce custom path? Or just on change (enter/blur)
            customPathWidget.callback = () => {
                if (folderWidget.value === "Custom") {
                    refreshFiles();
                }
            }
        }

        // Initial fetch
        setTimeout(refreshFiles, 500);

        // FIX for duplicate image:
        // The standard LoadImage node widget DOES draw the image.
        // If we want to hide it, we can override the widget's draw method 
        // OR try to trick it.
        // One common hack: set `imgWidget.showImage = false` if supported, or...
        // Override computeSize to not reserve space for image?

        // Better yet: If the node is `RizzLoadImage`, let's try to HIDE the widget's internal image
        // by intercepting the image load?

        // Actually, looking at ComfyUI core (widgets.js):
        // `loadImageWidget` ... `draw` calls `ctx.drawImage`.
        // We can override that specific instance's draw.

        if (imgWidget) {
            // Fix: Force the widget to display as a standard combo (text only)
            // 1. Override computeSize to return standard small height
            imgWidget.computeSize = function (width) {
                return [width, 30]; // Increased from 22 to 30 for visibility
            };

            // 2. Override draw to only draw the text/background
            const originalDraw = imgWidget.draw;
            imgWidget.draw = function (ctx, node, widgetWidth, y, widgetHeight) {
                const savedImage = this.image;
                this.image = null; // Hide the image so standard draw routine renders text
                try {
                    originalDraw.apply(this, arguments);
                } finally {
                    this.image = savedImage; // Restore so filtering logic works
                }
            };
        }
    }
}
