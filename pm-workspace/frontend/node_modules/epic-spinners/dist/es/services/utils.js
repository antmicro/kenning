var utils = {
  appendKeyframes: (name, frames) => {
    if (!window?.document) {
      return;
    }
    const sheet = document.createElement("style");
    sheet.setAttribute("id", name);
    sheet.innerHTML = `@keyframes ${name} {${frames}}`;
    document.body.appendChild(sheet);
  },
  removeKeyframes: (name) => {
    if (!window?.document) {
      return;
    }
    const sheet = document.getElementById(name);
    if (!sheet) {
      return;
    }
    sheet.parentNode?.removeChild(sheet);
  }
};

export { utils };
