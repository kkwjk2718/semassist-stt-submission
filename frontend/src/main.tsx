import { createRoot } from "react-dom/client";

import { App } from "./App";

const rootNode = document.getElementById("root");

if (rootNode === null) {
  throw new Error("Root element is missing");
}

createRoot(rootNode).render(<App />);
