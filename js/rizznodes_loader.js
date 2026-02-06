import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

console.log("[RizzNodes] File loader extension registered");

app.registerExtension({
    name: "RizzNodes.DynamicFileLoader",
    async nodeCreated(node) {
        if (node.comfyClass !== "RizzLoadAudio" && node.comfyClass !== "RizzLoadVideo" && node.comfyClass !== "RizzLoadImage") return;

        let type = "audio";
        if (node.comfyClass === "RizzLoadVideo") type = "video";
        if (node.comfyClass === "RizzLoadImage") type = "image";

        console.log(`[RizzNodes] Setting up dynamic ${type} combo for ${node.comfyClass}`);

        const fetchFiles = async (folder) => {
            if (!folder?.trim()) return [];
            try {
                const resp = await api.fetchApi("/rizz/list_files", {
                    method: "POST",
                    body: JSON.stringify({ path: folder, type: type }),
                });
                const data = await resp.json();
                return data.files || [];
            } catch (err) {
                console.error("[RizzNodes] Fetch error:", err);
                return [];
            }
        };

        const updateFileWidget = (files) => {
            const widget = node.widgets.find(w => w.name === "image");
            if (!widget) return;

            const values = files && files.length ? files : [""];
            widget.options.values = values;

            if (!values.includes(widget.value)) {
                widget.value = files && files.length > 0 ? files[0] : "";
            }

            node.setDirtyCanvas(true, true);
            updatePreview();
        };

        const folderWidget = node.widgets.find(w => w.name === "folder");
        const customPathWidget = node.widgets.find(w => w.name === "custom_path");

        const updateFiles = async () => {
            let path = "RizzImage"; // Default None

            if (folderWidget.value === "Custom") {
                path = customPathWidget.value;
                // Show custom path
                customPathWidget.type = customPathWidget.origType || "STRING";
            } else {
                if (folderWidget.value !== "None") {
                    path = "RizzImage/" + folderWidget.value;
                }
                // Hide custom path
                if (!customPathWidget.origType) customPathWidget.origType = customPathWidget.type;
                customPathWidget.type = "tschide";
            }

            // trigger resize if visibility changed
            node.setSize(node.computeSize());

            // Fetch files
            const files = await fetchFiles(path);
            updateFileWidget(files);
        };

        if (folderWidget) {
            folderWidget.callback = () => {
                updateFiles();
            };
        }

        if (customPathWidget) {
            // Cache original type
            customPathWidget.origType = customPathWidget.type;
            // Hide initially if not 'Custom'
            if (folderWidget.value !== "Custom") {
                customPathWidget.type = "tschide";
            }

            // Update on enter?
            customPathWidget.callback = () => {
                if (folderWidget.value === "Custom") updateFiles();
            };
        }

        node.addWidget("button", "Refresh Files", null, () => {
            updateFiles();
        });

        // Initial update
        setTimeout(() => {
            updateFiles();
        }, 500);

        // Preview Logic
        const imageWidget = node.widgets.find(w => w.name === "image");
        if (imageWidget && imageWidget.value === "None") {
            imageWidget.value = "";
        }

        const getWidgetBottom = () => {
            if (!node.widgets) return 0;
            return node.widgets.reduce((acc, w) => {
                let h = 0;
                if (w.computedHeight != null) h = w.computedHeight;
                else if (w.height != null) h = w.height;
                else if (w.computeSize) h = w.computeSize(node.size[0])[1];
                else h = 20;
                const y = w.y || 0;
                return Math.max(acc, y + h);
            }, 0);
        };

        const ensureInputCopy = async (filename) => {
            if (!filename || filename === "None") return;
            try {
                await api.fetchApi("/rizz/ensure_input", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        filename: filename,
                        folder: folderWidget?.value ?? "None",
                        custom_path: customPathWidget?.value ?? "",
                        type: type
                    })
                });
            } catch (err) {
                // Non-fatal; preview will still attempt output path
                console.warn("[RizzNodes] ensure_input failed:", err);
            }
        };

        function updatePreview() {
            if (!imageWidget || !imageWidget.value || imageWidget.value === "None") {
                node.rizz_img = null;
                return;
            }

            const filename = imageWidget.value;
            let folder_path = "RizzImage";
            if (folderWidget.value === "Custom") {
                folder_path = customPathWidget.value;
            } else if (folderWidget.value !== "None") {
                folder_path = "RizzImage/" + folderWidget.value;
            }

            // We need to support input dir (uploaded) or output dir (RizzImage)
            // We'll try fetching from OUTPUT first (since that's our main use case)
            // If that fails, we might try INPUT? Or simple check: drag/drop usually puts in 'input'
            // If the user selected from the dropdown which we populated, it's in folder_path (OUTPUT).
            // If they drag dropped, the value is just "filename.png" and it's in INPUT.
            // But we don't easily know WHICH 
            // Let's assume folder_path + filename in Output            
            const api_url = api.apiURL(`/view?filename=${encodeURIComponent(filename)}&subfolder=${encodeURIComponent(folder_path)}&type=output`);

            const img = new Image();
            img.onload = () => {
                node.rizz_img = img;
                // Auto-resize node to image dimensions
                // Add some padding for widgets
                const widgetHeight = getWidgetBottom() + 20;
                node.setSize([img.width, img.height + widgetHeight + 20]);
                node.setDirtyCanvas(true, true);
            };
            img.onerror = () => {
                // Try Input folder fallback (standard upload location)
                const fallback_url = api.apiURL(`/view?filename=${encodeURIComponent(filename)}&type=input`);
                const img2 = new Image();
                img2.onload = () => {
                    node.rizz_img = img2;
                    // Auto-resize node
                    const widgetHeight = getWidgetBottom() + 20;
                    node.setSize([img2.width, img2.height + widgetHeight + 20]);
                    node.setDirtyCanvas(true, true);
                };
                img2.src = fallback_url;
            };
            img.src = api_url;

            // Ensure ComfyUI input preview works for this file (optional)
            ensureInputCopy(filename);
        }

        // Hook into onDrawForeground to render the image
        const origOnDrawForeground = node.onDrawForeground;
        node.onDrawForeground = function (ctx) {
            if (origOnDrawForeground) origOnDrawForeground.apply(this, arguments);

            if (this.rizz_img) {
                const img = this.rizz_img;
                // Draw image to fill node or fit?

                const widgetHeight = getWidgetBottom() + 20;
                // Draw below widgets

                const ratio = img.width / img.height;
                const drawWidth = this.size[0] - 20;
                const drawHeight = drawWidth / ratio;

                // Ensure node is tall enough
                if (this.size[1] < widgetHeight + drawHeight + 20) {
                    this.size[1] = widgetHeight + drawHeight + 20;
                }

                ctx.drawImage(img, 10, widgetHeight, drawWidth, drawHeight);
            }
        };

        // Hide the widget's built-in image preview to avoid double rendering
        if (imageWidget) {
            imageWidget.computeSize = function (width) {
                return [width, 30];
            };

            const originalDraw = imageWidget.draw;
            if (typeof originalDraw === "function") {
                imageWidget.draw = function (ctx, node, widgetWidth, y, widgetHeight) {
                    const savedImage = this.image;
                    this.image = null;
                    try {
                        originalDraw.apply(this, arguments);
                    } finally {
                        this.image = savedImage;
                    }
                };
            }
        }

        // Update preview on changes
        if (imageWidget) {
            const origCallback = imageWidget.callback;
            imageWidget.callback = (v) => {
                ensureInputCopy(imageWidget.value);
                updatePreview();
                if (origCallback) origCallback.apply(this, arguments);
            };
        }

        const onConfigure = node.onConfigure;
        node.onConfigure = function () {
            if (onConfigure) onConfigure.apply(this, arguments);
            if (imageWidget && imageWidget.value === "None") {
                imageWidget.value = "";
            }
            updateFiles();
        };

        // Also update on refresh files
        const origUpdate = updateFiles;
        // We defined updateFiles as const, so we can't easily hook it unless we defined it inside this scope (which we did).
        // Actually, updateFiles calls updateFileWidget.
        // We can just call updatePreview() inside the button callback or after updateFiles resolves.
        // We already have: node.addWidget("button", "Refresh Files", ..., () => { updateFiles(); });
        // Let's check updatePreview triggers when widget value changes via code? 
        // Usually modifying .value directly doesn't trigger callback.
        // So we might need to manually call it.

        // Better: Hook updatePreview into updateFileWidget or call it after setting value.
        // But updateFileWidget is defined above.
        // We can just add a listener or call updatePreview() periodically? No.

        // Let's just run updatePreview once initially and rely on callback.
        setTimeout(() => updatePreview(), 600);
    }
});
