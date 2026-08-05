const baselineSystems = [
  ["gt", "Ground Truth"],
  ["dac", "DAC"],
  ["bigcodec", "BigCodec"],
  ["fmelcodec", "FMelCodec"],
  ["semanticodec", "SemantiCodec"],
  ["focalcodec", "FocalCodec"],
  ["umelcodec", "UMelCodec"]
];

const ablationSystems = [
  ["gt", "Ground Truth"],
  ["umelcodec", "UMelCodec"],
  ["no-msmd", "w/o MSMD"],
  ["no-af", "w/o AF filtering"],
  ["no-vq-factorization", "w/o VQ factorization"]
];

const codebookSystems = [
  ["gt", "Ground Truth"],
  ["1024", "K = 1024"],
  ["2048", "K = 2048"],
  ["4096", "K = 4096"],
  ["8192", "K = 8192"],
  ["16384", "K = 16384"]
];

function audioTable(root, systems, count, highlightedKeys = []) {
  const headers = systems.map(([key, label]) =>
    `<th class="${highlightedKeys.includes(key) ? "highlight" : ""}">${label}</th>`
  ).join("");
  const rows = Array.from({ length: count }, (_, index) => {
    const number = String(index + 1).padStart(2, "0");
    const cells = systems.map(([key, label]) => `
      <td>
        <audio controls preload="none" aria-label="${label}, sample ${number}">
          <source src="${root}/${number}/${key}.wav" type="audio/wav">
        </audio>
      </td>`).join("");
    return `<tr><td>${number}</td>${cells}</tr>`;
  }).join("");
  return `<table class="audio-table"><thead><tr><th>Sample</th>${headers}</tr></thead><tbody>${rows}</tbody></table>`;
}

function renderBaselines(split) {
  document.querySelector("#baseline-table").innerHTML = audioTable(
    `audio/baselines/test-${split}`,
    baselineSystems,
    6,
    ["umelcodec"]
  );
}

document.querySelectorAll("[data-baseline-split]").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll("[data-baseline-split]").forEach((item) => {
      const active = item === button;
      item.classList.toggle("active", active);
      item.setAttribute("aria-selected", String(active));
    });
    renderBaselines(button.dataset.baselineSplit);
  });
});

renderBaselines("clean");
document.querySelector("#ablation-table").innerHTML = audioTable(
  "audio/ablations/test-clean",
  ablationSystems,
  6,
  ["umelcodec"]
);
document.querySelector("#codebook-table").innerHTML = audioTable(
  "audio/codebooks/test-clean",
  codebookSystems,
  6,
  ["8192"]
);
