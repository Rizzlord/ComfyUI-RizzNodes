import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "rizznodes.texture_preview",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "RizzPreviewTiling") {
            // Hijack the onExecuted to get the image
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                if (onExecuted) onExecuted.apply(this, arguments);

                if (message?.images && message.images.length > 0) {
                    const imgParams = message.images[0];
                    const url = api.apiURL(`/view?filename=${imgParams.filename}&type=${imgParams.type}&subfolder=${imgParams.subfolder}`);

                    // Load image
                    this.previewImage = new Image();
                    this.previewImage.onload = () => {
                        this.setDirtyCanvas(true);
                    };
                    this.previewImage.src = url;
                }
            };

            // Custom draw 
            const onDrawForeground = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function (ctx) {
                if (onDrawForeground) onDrawForeground.apply(this, arguments);

                // Aggressively hide any widgets that might have been added (like the default image preview)
                if (this.widgets) {
                    let changed = false;
                    for (const w of this.widgets) {
                        // We strictly want NO widgets on this node, as it's just a 2x2 preview
                        if (w.type !== "hidden") {
                            w.type = "hidden";
                            w.computeSize = () => [0, -4];
                            changed = true;
                        }
                    }
                    // Only force resize if we actually hid something to avoid infinite loops
                    if (changed) {
                        // Defer resize to next tick to avoid layout thrashing during draw
                        setTimeout(() => {
                            this.setSize(this.computeSize());
                        }, 0);
                    }
                }

                if (!this.previewImage || !this.previewImage.complete || this.previewImage.naturalWidth === 0) {
                    return;
                }

                const w = this.size[0];
                const h = this.size[1];

                const margin = 10;
                const drawY = 30; // Skip header roughly
                const drawH = h - drawY - margin;
                const drawW = w - margin * 2;

                if (drawH < 20) return;

                ctx.save();

                // Clip to the draw area
                ctx.beginPath();
                ctx.rect(margin, drawY, drawW, drawH);
                ctx.clip();

                // Draw 2x2 grid (4 images total)
                const tileW = drawW / 2;
                const tileH = drawH / 2;

                for (let row = 0; row < 2; row++) {
                    for (let col = 0; col < 2; col++) {
                        ctx.drawImage(this.previewImage,
                            margin + col * tileW,
                            drawY + row * tileH,
                            tileW, tileH);
                    }
                }

                ctx.restore();
            };
        }
    }
});
