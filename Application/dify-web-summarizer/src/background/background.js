importScripts('lib/apiUtils.js');

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "callAPI") {
    chrome.storage.sync.get('apiKey', (data) => {
      if (data.apiKey) {
        callDifyAPI(data.apiKey, message.inputUrl)
          .then((response) => {
            if (response && response.data && response.data.outputs && response.data.outputs.text) {
              sendResponse({ data: response.data.outputs.text });
            } else {
              sendResponse({ error: "API response does not contain the expected data" });
            }
          })
          .catch((error) => sendResponse({ error: error.message }));
      } else {
        sendResponse({ error: "API Key is not set" });
      }
    });
    return true; // 保持消息通道開啟
  }
});