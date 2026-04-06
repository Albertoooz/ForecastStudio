# Jak używać agentów w Cursorze - Praktyczny Przewodnik

## Szybki Start (3 kroki)

### 1. Otwórz prompt agenta
```bash
# W terminalu Cursora
cat .prompts/architect.md
```

Lub po prostu otwórz plik `.prompts/architect.md` w edytorze.

### 2. Skopiuj całą zawartość
Zaznacz wszystko (Cmd+A / Ctrl+A) i skopiuj (Cmd+C / Ctrl+C).

### 3. Wklej do Cursor Chat
- Otwórz Cursor Chat (Cmd+L / Ctrl+L)
- Wklej prompt jako pierwsza wiadomość
- Dodaj swoje zadanie jako druga wiadomość

---

## Szczegółowy Workflow

### Przykład 1: Review Architektury

**Krok 1:** Otwórz `.prompts/architect.md` w Cursorze

**Krok 2:** Skopiuj całą zawartość

**Krok 3:** Otwórz Cursor Chat (Cmd+L)

**Krok 4:** Wklej prompt i dodaj zadanie:
```
[Wklej tutaj całą zawartość architect.md]

Review the current repository structure. Focus on forecaster/interface/conversation.py
```

**Krok 5:** Cursor przeanalizuje kod i da rekomendacje

**Krok 6:** Popraw kod zgodnie z rekomendacjami

---

### Przykład 2: Code Quality Check

**Krok 1:** Otwórz `.prompts/code_quality.md`

**Krok 2:** Skopiuj zawartość

**Krok 3:** W Cursor Chat:
```
[Wklej code_quality.md]

Review forecaster/models/simple.py for code quality issues.
Check naming, complexity, and style.
```

---

### Przykład 3: Test Coverage

**Krok 1:** Otwórz `.prompts/test_agent.md`

**Krok 2:** Skopiuj zawartość

**Krok 3:** W Cursor Chat:
```
[Wklej test_agent.md]

What tests are missing for forecaster/data/loader.py?
Identify edge cases and failure scenarios.
```

---

## Workflow: Pełny Cykl Development

### Scenariusz: Dodajesz nową funkcję

#### 1. Napisz kod
```python
# forecaster/models/new_model.py
class NewForecaster(BaseForecaster):
    ...
```

#### 2. Architect Agent Review
```
[Wklej architect.md]

Review forecaster/models/new_model.py for architectural issues.
Is this the simplest possible design?
```

**Popraw** zgodnie z rekomendacjami.

#### 3. Code Quality Agent
```
[Wklej code_quality.md]

Review forecaster/models/new_model.py for code quality.
Check naming, function length, clarity.
```

**Popraw** style issues.

#### 4. Test Agent
```
[Wklej test_agent.md]

What tests should I write for NewForecaster?
Identify edge cases and failure scenarios.
```

**Napisz** testy.

#### 5. Product Agent (opcjonalnie)
```
[Wklej product_agent.md]

Review how NewForecaster integrates with the user workflow.
Are there any hidden automation or missing control points?
```

---

## Zaawansowane: Używanie z @-mentions

Cursor pozwala na @-mentions plików. Możesz:

### Opcja A: Prompt + @file
```
[Wklej architect.md]

Review @forecaster/interface/conversation.py for architectural issues
```

### Opcja B: Prompt + @folder
```
[Wklej code_quality.md]

Review all files in @forecaster/models/ for code quality
```

### Opcja C: Prompt + @codebase
```
[Wklej architect.md]

Review the entire codebase structure.
Are there any unnecessary abstractions or layers?
```

---

## Pro Tips

### 1. Zapisz prompty jako snippets
W Cursorze możesz zapisać często używane prompty jako snippets:
- Settings → Snippets
- Dodaj snippet z promptem agenta
- Użyj skrótu klawiszowego

### 2. Używaj wielu agentów równolegle
Możesz otworzyć kilka chatów jednocześnie:
- Chat 1: Architect Agent
- Chat 2: Code Quality Agent
- Chat 3: Test Agent

Każdy z własnym promptem i zadaniem.

### 3. Kombinuj z foundation.md
Dla pełnego kontekstu, możesz wkleić najpierw foundation.md:

```
[Wklej foundation.md]

[Wklej architect.md]

Review the repository against these principles
```

### 4. Używaj konkretnych pytań
Zamiast:
```
Review the code
```

Używaj:
```
Review forecaster/models/automl.py.
Specifically check:
1. Is select_best_model() too complex?
2. Should evaluate_model() be split?
3. Are there unnecessary abstractions?
```

---

## Przykładowe Komendy

### Architect Agent
```
[architect.md]

Review @forecaster/agents/ for unnecessary abstractions
```

### Code Quality Agent
```
[code_quality.md]

Review @forecaster/models/simple.py line 42-46.
The date handling logic seems complex - can it be simplified?
```

### Test Agent
```
[test_agent.md]

What edge cases should I test for load_time_series()?
Consider: empty files, missing columns, invalid dates
```

### Product Agent
```
[product_agent.md]

Review the user workflow in @forecaster/main.py.
Are all control points explicit? Can users override decisions?
```

### Research Agent
```
[research_agent.md]

I need to add time series decomposition.
Research: statsmodels vs scipy vs custom implementation.
Consider maintenance and simplicity.
```

---

## Troubleshooting

### Problem: Cursor nie rozumie kontekstu
**Rozwiązanie:** Użyj @-mentions:
```
[prompt] Review @forecaster/models/base.py
```

### Problem: Zbyt długie odpowiedzi
**Rozwiązanie:** Bądź bardziej konkretny:
```
[prompt] Review only the BaseForecaster class definition (lines 22-55)
```

### Problem: Agent sugeruje zbyt dużo zmian
**Rozwiązanie:** Poproś o priorytetyzację:
```
[prompt] Review and prioritize: what are the top 3 most critical issues?
```

---

## Best Practices

1. **Jeden agent na raz** - nie mieszaj ról w jednym chacie
2. **Konkretne zadania** - "Review X for Y" zamiast "Review everything"
3. **Iteracyjnie** - najpierw Architect, potem Code Quality, potem Test
4. **Zapisuj rekomendacje** - kopiuj output agenta do notatek
5. **Weryfikuj zmiany** - po każdej poprawce, sprawdź czy nie zepsułeś czegoś

---

## Szybkie Referencje

| Agent | Kiedy użyć | Przykład |
|-------|-----------|----------|
| Architect | Po napisaniu kodu | "Review @forecaster/models/ for over-engineering" |
| Code Quality | Przed commit | "Review @forecaster/interface/ for naming issues" |
| Test | Przed merge | "What tests for @forecaster/data/loader.py?" |
| Product | Przed deploy | "Review user workflow in @forecaster/main.py" |
| Research | Przed dodaniem dependency | "Compare X vs Y for feature Z" |

---

## Integracja z Cursor 2.0 Multi-Agent

Cursor 2.0 oferuje Parallel Agents, Multi-Agent Judging i Plan Mode.

**📖 Zobacz [CURSOR_2.0_INTEGRATION.md](./CURSOR_2.0_INTEGRATION.md) dla szczegółów jak połączyć nasze prompty z funkcjami Cursor 2.0.**

**Szybkie podsumowanie:**
- Możesz używać naszych promptów z Parallel Agents (równoległe chaty)
- Możesz stworzyć custom commands (`/architect`, `/quality`) z naszymi promptami
- Plan Mode działa świetnie z Architect Agent promptem
- **Zalecenie:** Start manual, scale parallel, automate last

## Automatyzacja (przyszłość)

**NIE automatyzuj teraz.**

Dopiero gdy:
- ✅ Masz 2-3 pełne workflow
- ✅ Architektura jest stabilna
- ✅ Testy są w miejscu

Wtedy możesz rozważyć:
- Pre-commit hooks z agentami
- CI/CD integration
- Automatyczne review przed merge

Ale na razie: **manual control = better control**
