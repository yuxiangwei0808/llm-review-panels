const viewer = document.querySelector("#figure-viewer");
const viewport = document.querySelector("#figure-viewport");
const figureImage = document.querySelector("#figure-image");
const titles = {
  "pipeline.svg": "Four review stages and their failure modes",
  "scaling.svg": "Scoring power as the committee grows",
  "missed-weaknesses.svg": "Human-raised weaknesses missed by every review",
  "case-study.svg": "One rejected submission through the four stages",
  "remedy-effects.svg": "How the remedies change scores",
};
let trigger,
  ratio = 1,
  fittedWidth = 1,
  scale = 1;

function setScale(next, resetPosition = false) {
  const previousWidth =
    figureImage.getBoundingClientRect().width || fittedWidth;
  const centerX =
    (viewport.scrollLeft + viewport.clientWidth / 2) / previousWidth;
  const centerY =
    (viewport.scrollTop + viewport.clientHeight / 2) / (previousWidth / ratio);
  scale = Math.max(0.5, Math.min(4, next));
  const width = fittedWidth * scale;
  figureImage.style.width = `${width}px`;
  document.querySelector("#figure-scale").value = `${Math.round(scale * 100)}%`;
  document.querySelector("#figure-smaller").disabled = scale <= 0.5;
  document.querySelector("#figure-larger").disabled = scale >= 4;
  viewport.scrollLeft = resetPosition
    ? 0
    : centerX * width - viewport.clientWidth / 2;
  viewport.scrollTop = resetPosition
    ? 0
    : (centerY * width) / ratio - viewport.clientHeight / 2;
}

function fitFigure() {
  fittedWidth = Math.min(
    viewport.clientWidth - 32,
    (viewport.clientHeight - 32) * ratio,
  );
}

document.addEventListener("click", (event) => {
  const link = event.target.closest('a[href^="figures/"][href$=".svg"]');
  if (
    !link ||
    !viewer.showModal ||
    event.ctrlKey ||
    event.metaKey ||
    event.shiftKey ||
    event.altKey ||
    event.button !== 0
  )
    return;
  event.preventDefault();
  const original = link.closest("figure").querySelector("img");
  trigger = link;
  ratio =
    (original.naturalWidth || Number(original.getAttribute("width"))) /
    (original.naturalHeight || Number(original.getAttribute("height")));
  figureImage.src = link.href;
  figureImage.alt = original.alt;
  document.querySelector("#figure-viewer-title").textContent =
    titles[link.getAttribute("href").split("/").pop()] || "Paper figure";
  viewer.showModal();
  document.body.classList.add("viewer-open");
  fitFigure();
  // Start with readable lettering on phones; the figure remains a vector.
  setScale(innerWidth <= 700 ? Math.min(4, 1000 / fittedWidth) : 1, true);
});
document
  .querySelector("#figure-close")
  .addEventListener("click", () => viewer.close());
document
  .querySelector("#figure-smaller")
  .addEventListener("click", () => setScale(scale / 1.4));
document
  .querySelector("#figure-larger")
  .addEventListener("click", () => setScale(scale * 1.4));
document
  .querySelector("#figure-fit")
  .addEventListener("click", () => setScale(1, true));
viewer.addEventListener("close", () => {
  document.body.classList.remove("viewer-open");
  trigger?.focus({ preventScroll: true });
});
viewer.addEventListener("click", (event) => {
  const box = viewer.getBoundingClientRect();
  if (
    event.target === viewer &&
    (event.clientX < box.left ||
      event.clientX > box.right ||
      event.clientY < box.top ||
      event.clientY > box.bottom)
  )
    viewer.close();
});
window.addEventListener("resize", () => {
  if (!viewer.open) return;
  fitFigure();
  setScale(scale, true);
});

const copyButton = document.querySelector("#copy-bibtex");
copyButton.addEventListener("click", async () => {
  const text = document.querySelector("#bibtex").textContent;
  try {
    await navigator.clipboard.writeText(text);
    copyButton.textContent = "Copied";
  } catch {
    const range = document.createRange();
    range.selectNodeContents(document.querySelector("#bibtex"));
    getSelection().removeAllRanges();
    getSelection().addRange(range);
    copyButton.textContent = "Selected; press Ctrl+C";
  }
  setTimeout(() => (copyButton.textContent = "Copy BibTeX"), 2000);
});
