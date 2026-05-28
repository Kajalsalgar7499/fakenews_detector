const textArea = document.querySelector("#news_text");
const wordCount = document.querySelector("#word_count");

if (textArea && wordCount) {
    const updateCount = () => {
        const words = textArea.value.trim().split(/\s+/).filter(Boolean);
        wordCount.textContent = `${words.length} ${words.length === 1 ? "word" : "words"}`;
    };

    textArea.addEventListener("input", updateCount);
    updateCount();
}
