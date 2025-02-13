document.getElementById("summarize-btn").addEventListener("click", () => {
  chrome.storage.sync.get("apiKey", (data) => {
    const apiKey = data.apiKey;
    
    if (!apiKey) {
      document.getElementById("status").innerText = "Error: No API Key found.";
      return;
    }

    // update status and summary fields
    document.getElementById("status").innerText = "Summarizing...";
    document.getElementById("summary").innerText = "";

    // update URL field to the active tab's URL
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      const inputUrl = tabs[0]?.url;
      if (!inputUrl) {
        document.getElementById("status").innerText = "Error: Unable to retrieve active tab URL.";
        return;
      }
      
      chrome.runtime.sendMessage(
        { action: "callAPI", apiKey: apiKey, inputUrl: inputUrl },
        (response) => {
          if (response.error) {
            document.getElementById("status").innerText = `Error: ${response.error}`;
          } else {
            document.getElementById("status").innerText = `Summary:`;
            document.getElementById("summary").innerText = response.data;
          }
        }
      );
    });
  });
});

window.onload = () => {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const urlField = document.getElementById('url');
    urlField.value = tabs[0]?.url || window.location.href;
  });
};