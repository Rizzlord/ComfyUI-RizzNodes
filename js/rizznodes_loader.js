import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

console.log("[RizzNodes] File loader extension registered");

app.registerExtension({
    name: "RizzNodes.DynamicFileLoader",
    async nodeCreated(node) {
        if (node.comfyClass !== "RizzLoadAudio" && node.comfyClass !== "RizzLoadVideo") return;

        const type = node.comfyClass === "RizzLoadAudio" ? "audio" : "video";
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
            const widget = node.widgets.find(w => w.name === "file");
            if (!widget) return;

            const values = ["None", ...files];
            widget.options.values = values;

            if (!values.includes(widget.value)) {
                widget.value = "None";
            }

            node.setSize(node.computeSize());
            node.setDirtyCanvas(true, true);
        };

        const folderWidget = node.widgets.find(w => w.name === "folder_path");

        node.addWidget("button", "Scan Folder", null, async () => {
            if (!folderWidget?.value?.trim()) {
                console.warn("[RizzNodes] No folder path entered");
                return;
            }
            const files = await fetchFiles(folderWidget.value);
            updateFileWidget(files);
        });

        // Initial empty state
        if (node.widgets.find(w => w.name === "file")?.options?.values?.length <= 1) {
            updateFileWidget([]);
        }
    }
});
