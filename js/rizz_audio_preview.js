import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "rizznodes.audio_preview",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "RizzPreviewAudio") {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                if (onExecuted) onExecuted.apply(this, arguments);

                if (message?.audio && message.audio.length > 0) {
                    const audioParams = message.audio[0];
                    const url = api.apiURL(`/view?filename=${audioParams.filename}&type=${audioParams.type}&subfolder=${audioParams.subfolder}`);

                    if (!this.audioElement) {
                        const container = document.createElement("div");
                        container.style.padding = "4px 8px";
                        container.style.background = "rgba(10, 10, 10, 0.4)";
                        container.style.borderRadius = "12px";
                        container.style.display = "flex";
                        container.style.alignItems = "center";
                        container.style.justifyContent = "center";
                        container.style.marginTop = "-3px"; // Pushed up 15px from previous 12px
                        container.style.marginBottom = "8px";
                        container.style.border = "1px solid rgba(255, 255, 255, 0.1)";
                        container.style.boxShadow = "0 4px 12px rgba(0, 0, 0, 0.3)";

                        this.audioElement = document.createElement("audio");
                        this.audioElement.controls = true;
                        this.audioElement.style.width = "100%";
                        this.audioElement.style.height = "36px";

                        // Apply custom "Rizz" style to the default player via CSS filters
                        // This makes it dark and consistent with a premium UI
                        this.audioElement.style.filter = "invert(100%) hue-rotate(180deg) brightness(1.2) contrast(0.9)";
                        this.audioElement.style.opacity = "0.9";

                        container.appendChild(this.audioElement);

                        // Add as a DOM widget
                        this.addDOMWidget("rizz_audio_preview", "audio", container);

                        // Manual size adjustment - increased height for more bottom space
                        this.setSize([this.size[0], 150]);
                    }

                    this.audioElement.src = url;
                    this.audioElement.autoplay = !!audioParams.autoplay;
                    this.audioElement.loop = !!audioParams.loop;

                    if (this.audioElement.autoplay) {
                        // Small delay to ensure browser allows play
                        setTimeout(() => {
                            this.audioElement.play().catch(e => console.log("[RizzNodes] Autoplay prevented:", e));
                        }, 50);
                    }
                }
            };
        }
    }
});
