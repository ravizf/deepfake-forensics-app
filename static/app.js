document.addEventListener("DOMContentLoaded", () => {
  const input = document.querySelector("[data-preview-target]");
  if (!input) {
    return;
  }

  const previewShell = document.getElementById(input.dataset.previewTarget);
  input.addEventListener("change", () => {
    previewShell.innerHTML = "<h2>Preview</h2>";
    const file = input.files && input.files[0];
    if (!file) {
      previewShell.insertAdjacentHTML(
        "beforeend",
        "<p class='muted'>A preview appears here before upload so you can verify the submitted evidence.</p>"
      );
      return;
    }

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
});
