# 📚 Data Sources & Download Guide

This guide provides ready-to-use data sources for LLM training based on topics and keywords.

## 🎯 Quick Start

```bash
# Download sample corpus
python data/download_data.py --corpus

# Download specific topic
python data/download_data.py --topic technology --count 5

# Download specific books
python data/download_data.py --book 1342 --book 84 --book 11
```

## 📖 Available Topics

### Technology & Programming
- **Sources**: GitHub repositories, Stack Overflow, technical blogs
- **Formats**: Markdown, text files, code comments
- **Topics**: Python, JavaScript, AI/ML, web development, databases, DevOps

### Science & Mathematics
- **Sources**: Research papers, arXiv, scientific journals
- **Formats**: LaTeX, PDF, plain text
- **Topics**: Physics, chemistry, biology, mathematics, astronomy, medicine

### Literature & Arts
- **Sources**: Project Gutenberg, poetry collections, art criticism
- **Formats**: Plain text, poetry, novels, short stories
- **Topics**: Classic literature, modern fiction, poetry, drama, art history

### Philosophy & Ethics
- **Sources**: Classical texts, modern philosophy, ethics papers
- **Formats**: Plain text, philosophical essays
- **Topics**: Ethics, metaphysics, political philosophy, logic, epistemology

### History & Culture
- **Sources**: Historical documents, cultural studies, anthropology
- **Formats**: Plain text, historical accounts
- **Topics**: World history, ancient civilizations, cultural anthropology, archaeology

### Business & Economics
- **Sources**: Business reports, economic papers, market analysis
- **Formats**: PDF, text files, reports
- **Topics**: Economics, finance, marketing, management, entrepreneurship

### Health & Medicine
- **Sources**: Medical journals, health articles, wellness guides
- **Formats**: Plain text, medical documentation
- **Topics**: Medicine, nutrition, mental health, fitness, public health

### Law & Politics
- **Sources**: Legal documents, political speeches, government reports
- **Formats**: Plain text, legal documents
- **Topics**: Constitutional law, international law, political science, governance

### Education & Psychology
- **Sources**: Educational research, psychology papers, teaching materials
- **Formats**: Plain text, academic papers
- **Topics**: Educational psychology, pedagogy, cognitive science, child development

### Environment & Climate
- **Sources**: Environmental studies, climate research, sustainability reports
- **Formats**: Plain text, research papers
- **Topics**: Climate change, sustainability, environmental science, conservation

### Sports & Recreation
- **Sources**: Sports journalism, fitness guides, recreational activities
- **Formats**: Plain text, sports reporting
- **Topics**: Sports history, fitness training, outdoor activities, sports psychology

### Food & Culinary
- **Sources**: Recipe collections, culinary history, food science
- **Formats**: Plain text, recipes, food writing
- **Topics**: Cooking techniques, food history, nutrition, culinary culture

### Travel & Geography
- **Sources**: Travel guides, geographical studies, cultural exploration
- **Formats**: Plain text, travel writing
- **Topics**: World geography, travel narratives, cultural exploration, tourism

### Religion & Spirituality
- **Sources**: Religious texts, spiritual writings, theological studies
- **Formats**: Plain text, religious documents
- **Topics**: World religions, spirituality, theology, comparative religion

### Language & Linguistics
- **Sources**: Linguistic studies, language learning materials, etymology
- **Formats**: Plain text, linguistic analysis
- **Topics**: Language evolution, linguistics, grammar, language learning

### Media & Journalism
- **Sources**: News articles, media analysis, journalism studies
- **Formats**: Plain text, news reports
- **Topics**: Media studies, journalism ethics, digital media, communications
- **Topics**: World history, ancient civilizations

## 📊 Sample Data Sizes

| Topic | Files | Avg Size | Total Size |
|-------|-------|----------|------------|
| Literature | 5 books | ~500KB | ~2.5MB |
| Technology | 3 docs | ~100KB | ~300KB |
| Science | 3 papers | ~200KB | ~600KB |

## 🔍 Popular Book IDs (Project Gutenberg)

### Literature Classics
- **1342** - Pride and Prejudice
- **84** - Frankenstein  
- **11** - Alice in Wonderland
- **1661** - Grimm's Fairy Tales
- **2701** - Moby Dick
- **64317** - Dracula

### Philosophy
- **3207** - Plato's Republic
- **13476** - Aristotle's Ethics
- **3200** - Marcus Aurelius Meditations

### History
- **19712** - History of Rome
- **1998** - Decline and Fall of Rome
- **28054** - World History

## 🛠️ Custom Downloads

### Adding New Sources
```python
# In download_data.py, add to topic_sources
topic_sources["new_topic"] = [
    "https://example.com/document1.txt",
    "https://example.com/document2.md",
]
```

### Direct URL Downloads
```python
# Download from any URL
python -c "
from data.download_data import SmartDataDownloader
d = SmartDataDownloader()
d.download_from_urls(['https://example.com/file.txt'], 'custom')
"
```

## 🎓 Training Tips

1. **Start Small**: Begin with 3-5 files (~1-2MB total)
2. **Mix Topics**: Combine literature + technology for diverse vocabulary
3. **Quality Check**: Review downloaded files for relevance
4. **Incremental**: Add more data gradually as you test

## 📈 Size Recommendations

- **Testing**: 1-5MB total data
- **Small Training**: 10-50MB total data  
- **Full Training**: 100MB+ total data

## 🔧 Troubleshooting

- **Download failures**: Check internet connection and URL validity
- **Large files**: Use `--count 1` for single large documents
- **Storage**: Monitor `data/raw/` directory size
- **Format issues**: All files converted to UTF-8 plain text
