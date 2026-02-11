import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function getPreviewEntry(message) {
    if (message?.videos?.length > 0) return message.videos[0];
    if (message?.video?.length > 0) return message.video[0];
    return null;
}

app.registerExtension({
    name: "rizznodes.video_preview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "RizzPreviewVideo") return;

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            // Strip image-related keys before calling the original handler
            // to prevent ComfyUI's core from creating a competing image preview.
            const cleaned = Object.assign({}, message);
            delete cleaned.images;
            delete cleaned.gifs;
            if (onExecuted) onExecuted.call(this, cleaned);

            const videoParams = getPreviewEntry(message);
            if (!videoParams?.filename) return;

            const query = new URLSearchParams({
                filename: videoParams.filename,
                type: videoParams.type || "temp",
                subfolder: videoParams.subfolder || "",
            });
            const url = api.apiURL(`/view?${query.toString()}`);

            if (!this.videoElement) {
                const container = document.createElement("div");
                container.style.padding = "6px 8px";
                container.style.background = "rgba(10, 10, 10, 0.4)";
                container.style.borderRadius = "12px";
                container.style.display = "flex";
                container.style.flexDirection = "column";
                container.style.alignItems = "center";
                container.style.justifyContent = "center";
                container.style.marginTop = "-3px";
                container.style.marginBottom = "8px";
                container.style.border = "1px solid rgba(255, 255, 255, 0.1)";
                container.style.boxShadow = "0 4px 12px rgba(0, 0, 0, 0.3)";

                this.videoElement = document.createElement("video");
                this.videoElement.controls = true;
                this.videoElement.style.width = "100%";
                this.videoElement.style.borderRadius = "8px";
                this.videoElement.playsInline = true;

                this.resolutionLabel = document.createElement("div");
                this.resolutionLabel.style.color = "rgba(255, 255, 255, 0.6)";
                this.resolutionLabel.style.fontSize = "11px";
                this.resolutionLabel.style.fontFamily = "monospace";
                this.resolutionLabel.style.marginTop = "4px";
                this.resolutionLabel.style.textAlign = "center";
                this.resolutionLabel.textContent = "";

                const resLabel = this.resolutionLabel;
                this.videoElement.addEventListener("loadedmetadata", function () {
                    const w = this.videoWidth;
                    const h = this.videoHeight;
                    resLabel.textContent = (w && h) ? `${w} × ${h}` : "";
                });

                container.appendChild(this.videoElement);
                container.appendChild(this.resolutionLabel);
                this.addDOMWidget("rizz_video_preview", "video", container);

                if (this.size[1] < 220) {
                    this.setSize([this.size[0], 220]);
                }
            }

            const autoplay = !!videoParams.autoplay;
            const loop = !!videoParams.loop;

            this.resolutionLabel.textContent = "";
            this.videoElement.src = url;
            this.videoElement.loop = loop;
            this.videoElement.autoplay = autoplay;
            this.videoElement.muted = autoplay;
            this.videoElement.load();

            if (autoplay) {
                const playPromise = this.videoElement.play();
                if (playPromise?.catch) {
                    playPromise.catch(() => {
                        // Browser autoplay policies may block playback.
                    });
                }
            }
        };
    },
});
