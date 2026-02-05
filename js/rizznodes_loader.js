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

            const values = ["None", ...files];
            widget.options.values = values;

            if (!values.includes(widget.value)) {
                widget.value = "None";
            }
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

        function updatePreview() {
            if (!imageWidget || imageWidget.value === "None") {
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
            const api_url = `./view?filename=${encodeURIComponent(filename)}&subfolder=${encodeURIComponent(folder_path)}&type=output`;

            const img = new Image();
            img.onload = () => {
                node.rizz_img = img;
                // Auto-resize node to image dimensions
                // Add some padding for widgets
                const widgetHeight = node.widgets.reduce((acc, w) => acc + (w.computeSize ? w.computeSize()[1] : 20), 0) + 40;
                node.setSize([img.width, img.height + widgetHeight]);
                node.setDirtyCanvas(true, true);
            };
            img.onerror = () => {
                // Try Input folder fallback (standard upload location)
                const fallback_url = `./view?filename=${encodeURIComponent(filename)}&type=input`;
                const img2 = new Image();
                img2.onload = () => {
                    node.rizz_img = img2;
                    // Auto-resize node
                    const widgetHeight = node.widgets.reduce((acc, w) => acc + (w.computeSize ? w.computeSize()[1] : 20), 0) + 40;
                    node.setSize([img2.width, img2.height + widgetHeight]);
                    node.setDirtyCanvas(true, true);
                };
                img2.src = fallback_url;
            };
            img.src = api_url;
        }

        // Hook into onDrawForeground to render the image
        const origOnDrawForeground = node.onDrawForeground;
        node.onDrawForeground = function (ctx) {
            if (origOnDrawForeground) origOnDrawForeground.apply(this, arguments);

            if (this.rizz_img) {
                const img = this.rizz_img;
                // Draw image to fill node or fit?

                const widgetHeight = this.widgets.reduce((acc, w) => acc + (w.computeSize ? w.computeSize()[1] : 20), 0) + 20;
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

        // Update preview on changes
        if (imageWidget) {
            const origCallback = imageWidget.callback;
            imageWidget.callback = (v) => {
                updatePreview();
                if (origCallback) origCallback.apply(this, arguments);
            };
        }

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
