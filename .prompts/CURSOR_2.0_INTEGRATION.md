# Integracja z Cursor 2.0 Multi-Agent

## Różnica: Nasz Setup vs Cursor 2.0

### Nasz Obecny Setup (Manual Multi-Agent)
- ✅ **Manual orchestration** - ty decydujesz kiedy którego agenta użyć
- ✅ **Prompty w plikach** - kopiujesz i wklejasz do chatu
- ✅ **Sekwencyjne** - jeden agent po drugim
- ✅ **Pełna kontrola** - każda decyzja jest twoja
- ❌ **Nie automatyczne** - musisz ręcznie wybierać agentów

### Cursor 2.0 Multi-Agent (Automatyczne)
- ✅ **Parallel Agents** - do 8 agentów równolegle
- ✅ **Multi-Agent Judging** - automatyczna ocena wyników
- ✅ **Subagents** - składnia `/name` do wywoływania
- ✅ **Plan Mode** - planowanie przed implementacją
- ❌ **Mniej kontroli** - automatyczna orkiestracja

## Jak Połączyć Oba Podejścia

### Opcja 1: Użyj Cursor 2.0 z Naszymi Promptami (Zalecane)

Możesz wykorzystać funkcje Cursor 2.0, ale z naszymi specjalistycznymi promptami.

#### Parallel Agents z Naszymi Promptami

**Przykład:** Równoległy review przez 3 agentów

1. Otwórz 3 osobne chaty w Cursorze
2. W każdym wklej inny prompt:

**Chat 1 - Architect Agent:**
```
[Wklej .prompts/architect.md]

Review @forecaster/models/ for architectural issues
```

**Chat 2 - Code Quality Agent:**
```
[Wklej .prompts/code_quality.md]

Review @forecaster/models/ for code quality issues
```

**Chat 3 - Test Agent:**
```
[Wklej .prompts/test_agent.md]

What tests are missing for @forecaster/models/?
```

3. Uruchom wszystkie 3 równolegle (Cursor 2.0 obsługuje to automatycznie)
4. Porównaj wyniki ręcznie (lub użyj Multi-Agent Judging)

#### Multi-Agent Judging

Po otrzymaniu wyników od wszystkich agentów, możesz poprosić o ocenę:

```
I got reviews from 3 agents:
- Architect Agent: [wklej wynik]
- Code Quality Agent: [wklej wynik]
- Test Agent: [wklej wynik]

Which issues should I prioritize? Give me top 5 most critical.
```

#### Plan Mode z Architect Agent

Przed implementacją, użyj Plan Mode z Architect promptem:

1. W Cursor Chat, włącz Plan Mode (Shift+Tab)
2. Wklej Architect prompt:
```
[Wklej .prompts/architect.md]

Create a plan for refactoring forecaster/models/automl.py
Focus on simplifying select_best_model()
```

3. Przejrzyj plan przed implementacją
4. Edytuj plan jeśli potrzeba
5. Dopiero potem implementuj

---

### Opcja 2: Subagents z Naszymi Promptami

Możesz stworzyć własne "subagenty" używając składni `/name` w Cursorze.

#### Konfiguracja Subagentów

W Cursorze możesz zdefiniować custom commands:

**Settings → Custom Commands → Add**

**Command 1: `/architect`**
```
[Wklej .prompts/architect.md]

Review the selected code for architectural issues
```

**Command 2: `/code-quality`**
```
[Wklej .prompts/code_quality.md]

Review the selected code for code quality issues
```

**Command 3: `/test-review`**
```
[Wklej .prompts/test_agent.md]

What tests are missing for the selected code?
```

**Command 4: `/product-review`**
```
[Wklej .prompts/product_agent.md]

Review the selected code for product/workflow issues
```

#### Użycie

Teraz możesz używać:
```
/architect @forecaster/models/simple.py
```

Zamiast kopiować prompt za każdym razem!

---

### Opcja 3: Hybrydowe Podejście (Najlepsze)

**Dla prostych zadań:** Użyj manual orchestration (nasz obecny setup)
- Szybkie, kontrolowane
- Jeden agent na raz
- Pełna kontrola

**Dla dużych zmian:** Użyj Cursor 2.0 Parallel Agents
- Równoległy review przez wielu agentów
- Automatyczna ocena wyników
- Szybsze dla dużych projektów

**Przykład workflow:**

1. **Mała zmiana** (np. fix buga):
   ```
   [Wklej code_quality.md]
   Review this fix: [kod]
   ```

2. **Duża zmiana** (np. refactoring całego modułu):
   - Uruchom 3 równoległe chaty:
     - Architect Agent
     - Code Quality Agent
     - Test Agent
   - Użyj Multi-Agent Judging do oceny
   - Wybierz najlepsze rekomendacje

---

## Konkretne Przykłady

### Przykład 1: Równoległy Review Feature

**Zadanie:** Dodałeś nowy model forecasting

**Krok 1:** Uruchom 3 równoległe chaty

**Chat 1:**
```
[.prompts/architect.md]

Review @forecaster/models/new_model.py
Is the design simple enough? Any unnecessary abstractions?
```

**Chat 2:**
```
[.prompts/code_quality.md]

Review @forecaster/models/new_model.py
Check naming, complexity, style
```

**Chat 3:**
```
[.prompts/test_agent.md]

What tests should I write for @forecaster/models/new_model.py?
Identify edge cases
```

**Krok 2:** Po otrzymaniu wszystkich odpowiedzi, poproś o ocenę:
```
I have 3 reviews for new_model.py:
- Architect: [wklej]
- Code Quality: [wklej]
- Test: [wklej]

Prioritize: what are the top 3 most critical issues to fix?
```

**Krok 3:** Napraw zgodnie z priorytetami

---

### Przykład 2: Plan Mode przed Refactoringiem

**Zadanie:** Chcesz uprościć `select_best_model()`

**Krok 1:** Włącz Plan Mode (Shift+Tab w chat)

**Krok 2:**
```
[.prompts/architect.md]

Create a detailed plan for refactoring forecaster/models/automl.py
Specifically: simplify select_best_model() function
Current issues: [wklej z ARCHITECTURE_REVIEW.md]
```

**Krok 3:** Przejrzyj plan, edytuj jeśli potrzeba

**Krok 4:** Zatwierdź plan

**Krok 5:** Implementuj zgodnie z planem

**Krok 6:** Po implementacji, review:
```
[.prompts/code_quality.md]

Review the refactored select_best_model()
Did we achieve the simplification goals?
```

---

### Przykład 3: Custom Subagents

**Setup (raz):**

1. Settings → Custom Commands
2. Dodaj:

**`/architect`**
```
[Zawartość .prompts/architect.md]

Review the selected code
```

**`/quality`**
```
[Zawartość .prompts/code_quality.md]

Review the selected code
```

**`/test`**
```
[Zawartość .prompts/test_agent.md]

Review the selected code
```

**Użycie:**

Teraz możesz po prostu:
```
/architect @forecaster/models/
```

Zamiast kopiować prompt za każdym razem!

---

## Rekomendacje

### Kiedy Używać Manual (Obecny Setup)
- ✅ Małe zmiany (1-2 pliki)
- ✅ Kiedy potrzebujesz pełnej kontroli
- ✅ Kiedy chcesz zrozumieć każdy krok
- ✅ Podczas nauki/eksperymentowania

### Kiedy Używać Cursor 2.0 Parallel
- ✅ Duże refactoringi (wiele plików)
- ✅ Kiedy potrzebujesz szybkiego feedbacku
- ✅ Kiedy masz doświadczenie z workflow
- ✅ Przy code review przed merge

### Kiedy Używać Plan Mode
- ✅ Przed dużymi zmianami
- ✅ Kiedy chcesz przejrzeć plan przed implementacją
- ✅ Dla złożonych refactoringów
- ✅ Kiedy potrzebujesz dokumentacji planu

---

## Migracja Stopniowa

**Faza 1 (Teraz):** Manual orchestration
- Używaj obecnego setupu
- Naucz się workflow
- Zrozum każdy agent

**Faza 2 (Po 2-3 tygodniach):** Dodaj Custom Commands
- Skonfiguruj `/architect`, `/quality`, `/test`
- Szybsze wywoływanie
- Nadal manual orchestration

**Faza 3 (Po miesiącu):** Eksperymentuj z Parallel
- Dla dużych zmian
- Porównaj wyniki z manual
- Zdecyduj co działa lepiej

**Faza 4 (Gdy gotowy):** Automatyzacja
- Pre-commit hooks
- CI/CD integration
- Ale tylko jeśli manual process jest sprawdzony

---

## Podsumowanie

**Mamy:** Manual multi-agent setup z specjalistycznymi promptami

**Cursor 2.0 oferuje:** Automatyczną orkiestrację i równoległą pracę

**Najlepsze podejście:**
- Użyj naszych promptów (są lepsze - specjalistyczne)
- Wykorzystaj funkcje Cursor 2.0 (szybsze, równoległe)
- Zachowaj kontrolę (manual dla małych, parallel dla dużych)

**Zasada:** Start manual, scale parallel, automate last.
