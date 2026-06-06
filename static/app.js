document.addEventListener("DOMContentLoaded", () => {
  const input = document.querySelector("[data-preview-target]");
  if (input) {
    const previewShell = document.getElementById(input.dataset.previewTarget);
    input.addEventListener("change", () => {
      previewShell.innerHTML = "<h2>Preview</h2>";
      const files = Array.from(input.files || []);
      if (!files.length) {
        previewShell.insertAdjacentHTML(
          "beforeend",
          "<p class='muted'>A preview appears here before upload so you can verify the submitted evidence.</p>"
        );
        return;
      }

      previewShell.insertAdjacentHTML(
        "beforeend",
        `<p class="muted">${files.length} file${files.length === 1 ? "" : "s"} selected.</p>`
      );

      files.slice(0, 4).forEach((file) => {
        const objectUrl = URL.createObjectURL(file);
        previewShell.insertAdjacentHTML(
          "beforeend",
          `<p><strong>${file.name}</strong><br><span class="muted">${Math.round(file.size / 1024)} KB</span></p>`
        );

        if (file.type.startsWith("video/")) {
          const video = document.createElement("video");
          video.src = objectUrl;
          video.controls = true;
          previewShell.appendChild(video);
        } else {
          const image = document.createElement("img");
          image.src = objectUrl;
          image.alt = "Evidence preview";
          previewShell.appendChild(image);
        }
      });

      if (files.length > 4) {
        previewShell.insertAdjacentHTML(
          "beforeend",
          `<p class="muted">Showing the first 4 previews. ${files.length - 4} more file(s) will still be analyzed.</p>`
        );
      }
    });
  }

  document.querySelectorAll("[data-loading-form]").forEach((form) => {
    form.addEventListener("submit", () => {
      const panel = form.querySelector("[data-loading-panel]");
      const button = form.querySelector("button[type='submit']");
      if (panel) {
        panel.hidden = false;
      }
      if (button) {
        button.disabled = true;
        button.textContent = "Analyzing...";
      }
    });
  });
});
