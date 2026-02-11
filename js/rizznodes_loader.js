import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

console.log("[RizzNodes] File loader extension registered");

const MIN_PREVIEW_WIDTH = 220;
const PREVIEW_MIN_HEIGHT = 320;
const PREVIEW_PADDING = 10;
const PREVIEW_TOP_GAP = 20;
const RECOVER_HEIGHT_LIMIT = 5000;

function getImageRatio(img) {
    const width = img?.naturalWidth || img?.width || 0;
    const height = img?.naturalHeight || img?.height || 0;
    if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
        return null;
    }
    const ratio = width / height;
    if (!Number.isFinite(ratio) || ratio <= 0) {
        return null;
    }
    return ratio;
}

function getImageDimensions(img) {
    const width = img?.naturalWidth || img?.width || 0;
    const height = img?.naturalHeight || img?.height || 0;
    if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
        return null;
    }
    return { width, height };
}

app.registerExtension({
    name: "RizzNodes.DynamicFileLoader",
    async nodeCreated(node) {
        if (node.comfyClass !== "RizzLoadAudio" && node.comfyClass !== "RizzLoadVideo" && node.comfyClass !== "RizzLoadImage") return;

        const isImageNode = node.comfyClass === "RizzLoadImage";
        let type = "audio";
        if (node.comfyClass === "RizzLoadVideo") type = "video";
        if (isImageNode) type = "image";

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

        let previewRequestToken = 0;
        const emptySelection = isImageNode ? "" : "None";
        const folderWidgetName = isImageNode ? "folder" : "folder_path";
        const fileWidgetName = isImageNode ? "image" : "file";
        const folderWidget = node.widgets.find(w => w.name === folderWidgetName);
        const customPathWidget = isImageNode ? node.widgets.find(w => w.name === "custom_path") : null;
        const fileWidget = node.widgets.find(w => w.name === fileWidgetName);

        const updateFileWidget = (files) => {
            if (!fileWidget) return;

            const values = files && files.length ? files : [emptySelection];
            fileWidget.options.values = values;

            if (!values.includes(fileWidget.value)) {
                fileWidget.value = files && files.length > 0 ? files[0] : emptySelection;
            }

            node.setDirtyCanvas(true, true);
            if (isImageNode) {
                updatePreview(fileWidget.value, false);
            }
        };

        const updateFiles = async () => {
            let path = type === "video" ? "RizzVideo" : type === "audio" ? "RizzAudio" : "RizzImage";

            if (isImageNode) {
                if (folderWidget?.value === "Custom") {
                    path = customPathWidget?.value ?? "";
                    // Show custom path
                    if (customPathWidget) {
                        customPathWidget.type = customPathWidget.origType || "STRING";
                    }
                } else {
                    if (folderWidget?.value && folderWidget.value !== "None") {
                        path = "RizzImage/" + folderWidget.value;
                    }
                    // Hide custom path
                    if (customPathWidget) {
                        if (!customPathWidget.origType) customPathWidget.origType = customPathWidget.type;
                        customPathWidget.type = "tschide";
                    }
                }
            } else if (folderWidget?.value) {
                path = folderWidget.value;
            }

            if (isImageNode) {
                // Expand node if visibility changed caused widgets to overflow
                const minSize = node.computeSize();
                if (node.size[0] < minSize[0] || node.size[1] < minSize[1]) {
                    node.setSize([Math.max(node.size[0], minSize[0]), Math.max(node.size[1], minSize[1])]);
                }
            }

            // Fetch files
            const files = await fetchFiles(path);
            updateFileWidget(files);
        };

        if (folderWidget) {
            folderWidget.callback = () => {
                updateFiles();
            };
        }

        if (customPathWidget && folderWidget) {
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

        if (!isImageNode) {
            const onConfigure = node.onConfigure;
            node.onConfigure = function () {
                if (onConfigure) onConfigure.apply(this, arguments);
                updateFiles();
            };
            return;
        }

        // Preview Logic
        const imageWidget = fileWidget;
        if (imageWidget && imageWidget.value === "None") {
            imageWidget.value = "";
        }

        const getWidgetBottom = () => {
            if (!node.widgets) return 0;
            const nodeWidth = Number.isFinite(node.size?.[0]) ? node.size[0] : 300;
            return node.widgets.reduce((acc, w) => {
                let h = 20;
                if (Number.isFinite(w.computedHeight)) {
                    h = w.computedHeight;
                } else if (Number.isFinite(w.height)) {
                    h = w.height;
                } else if (typeof w.computeSize === "function") {
                    const size = w.computeSize(nodeWidth);
                    if (Array.isArray(size) && Number.isFinite(size[1])) {
                        h = size[1];
                    }
                }
                const y = Number.isFinite(w.y) ? w.y : 0;
                const bottom = y + h;
                if (!Number.isFinite(bottom)) {
                    return acc;
                }
                return Math.max(acc, bottom);
            }, 0);
        };

        const ensurePreviewNodeSize = () => {
            const minSize = node.computeSize();
            const widgetBottom = getWidgetBottom();
            const desiredWidth = Math.max(node.size[0], minSize[0], MIN_PREVIEW_WIDTH);
            const minPreviewHeight = widgetBottom + PREVIEW_TOP_GAP + PREVIEW_MIN_HEIGHT + PREVIEW_PADDING;
            const currentHeight = Number.isFinite(node.size[1]) ? node.size[1] : 0;
            const desiredHeight = currentHeight > RECOVER_HEIGHT_LIMIT
                ? Math.max(minSize[1], minPreviewHeight)
                : Math.max(currentHeight, minSize[1], minPreviewHeight);
            if (node.size[0] !== desiredWidth || node.size[1] !== desiredHeight) {
                node.setSize([desiredWidth, desiredHeight]);
            }
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

        function updatePreview(selectedValue = imageWidget?.value, adjustSize = false) {
            if (!imageWidget || !selectedValue || selectedValue === "None") {
                previewRequestToken += 1;
                node.rizz_img = null;
                node.setDirtyCanvas(true, true);
                return;
            }

            const filename = selectedValue;
            const requestToken = ++previewRequestToken;
            const cacheBuster = Date.now();
            let folder_path = "RizzImage";
            if (folderWidget?.value === "Custom") {
                folder_path = customPathWidget?.value ?? "";
            } else if (folderWidget?.value && folderWidget.value !== "None") {
                folder_path = "RizzImage/" + folderWidget.value;
            }

            // We need to support input dir (uploaded) or output dir (RizzImage)
            // We'll try fetching from OUTPUT first (since that's our main use case)
            // If that fails, we might try INPUT? Or simple check: drag/drop usually puts in 'input'
            // If the user selected from the dropdown which we populated, it's in folder_path (OUTPUT).
            // If they drag dropped, the value is just "filename.png" and it's in INPUT.
            // But we don't easily know WHICH 
            // Let's assume folder_path + filename in Output            
            const api_url = api.apiURL(`/view?filename=${encodeURIComponent(filename)}&subfolder=${encodeURIComponent(folder_path)}&type=output&rizz_ts=${cacheBuster}`);

            const img = new Image();
            img.onload = () => {
                if (requestToken !== previewRequestToken) return;
                const ratio = getImageRatio(img);
                if (!ratio) {
                    console.warn(`[RizzNodes] Invalid image dimensions for preview: ${filename}`);
                    node.rizz_img = null;
                    node.setDirtyCanvas(true, true);
                    return;
                }
                node.rizz_img = img;
                // Keep preview area stable (BlenderTools pattern): do not size from image dimensions.
                if (adjustSize) {
                    ensurePreviewNodeSize();
                }
                node.setDirtyCanvas(true, true);
            };
            img.onerror = () => {
                if (requestToken !== previewRequestToken) return;
                // Try Input folder fallback (standard upload location)
                const fallback_url = api.apiURL(`/view?filename=${encodeURIComponent(filename)}&type=input&rizz_ts=${cacheBuster}`);
                const img2 = new Image();
                img2.onload = () => {
                    if (requestToken !== previewRequestToken) return;
                    const ratio = getImageRatio(img2);
                    if (!ratio) {
                        console.warn(`[RizzNodes] Invalid fallback image dimensions for preview: ${filename}`);
                        node.rizz_img = null;
                        node.setDirtyCanvas(true, true);
                        return;
                    }
                    node.rizz_img = img2;
                    if (adjustSize) {
                        ensurePreviewNodeSize();
                    }
                    node.setDirtyCanvas(true, true);
                };
                img2.onerror = () => {
                    if (requestToken !== previewRequestToken) return;
                    node.rizz_img = null;
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
                const widgetHeight = getWidgetBottom() + PREVIEW_TOP_GAP;
                const ratio = getImageRatio(img);
                if (!ratio) {
                    this.rizz_img = null;
                    this.setDirtyCanvas(true, true);
                    return;
                }

                const dims = getImageDimensions(img);
                if (!dims) {
                    return;
                }

                const previewWidth = Math.max(this.size[0] - (PREVIEW_PADDING * 2), 1);
                const previewHeight = Math.max(this.size[1] - widgetHeight - PREVIEW_PADDING, 1);
                if (previewWidth <= 0 || previewHeight <= 0) {
                    return;
                }

                // Fit image inside preview area without changing node size.
                const scale = Math.min(previewWidth / dims.width, previewHeight / dims.height);
                if (!Number.isFinite(scale) || scale <= 0) {
                    return;
                }
                const drawWidth = Math.max(1, dims.width * scale);
                const drawHeight = Math.max(1, dims.height * scale);
                const drawX = PREVIEW_PADDING + ((previewWidth - drawWidth) / 2);
                const drawY = widgetHeight + ((previewHeight - drawHeight) / 2);

                ctx.drawImage(img, drawX, drawY, drawWidth, drawHeight);
            }
        };

        // Hide built-in Comfy image preview for this node to prevent stale/duplicate images.
        const origOnDrawBackground = node.onDrawBackground;
        node.onDrawBackground = function (ctx) {
            const savedImgs = this.imgs;
            this.imgs = null;
            try {
                if (origOnDrawBackground) origOnDrawBackground.apply(this, arguments);
            } finally {
                this.imgs = savedImgs;
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
            imageWidget.callback = function (v) {
                if (origCallback) origCallback.apply(this, arguments);
                if (typeof v === "string" && v.length > 0) {
                    imageWidget.value = v;
                }
                node.imgs = null;
                ensureInputCopy(imageWidget.value);
                updatePreview(imageWidget.value, false);
            };
        }

        // Re-fetch preview after execution so overwritten files update in-place.
        const origOnExecuted = node.onExecuted;
        node.onExecuted = function () {
            if (origOnExecuted) origOnExecuted.apply(this, arguments);
            this.imgs = null;
            updatePreview(imageWidget?.value, false);
        };

        const onConfigure = node.onConfigure;
        node.onConfigure = function () {
            if (onConfigure) onConfigure.apply(this, arguments);
            if (imageWidget && imageWidget.value === "None") {
                imageWidget.value = "";
            }
            ensurePreviewNodeSize();
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

        // Initialize size once; later image switches should not resize.
        setTimeout(() => {
            ensurePreviewNodeSize();
            updatePreview(undefined, false);
        }, 600);
    }
});
