# LocalLLM Server (iPadOS)

Aplikacja na iPadOS uruchamiajaca lokalnie model GGUF (llama.cpp + Metal) i wystawiajaca
w sieci lokalnej serwer **zgodny z OpenAI Chat Completions**. Powstala do jednego celu:
zmierzyc wydajnosc `gemma-3-4b-it` Q4_K_M na Apple Silicon — prefill, generacja, szczyt
pamieci, dlawienie termiczne — na realnym ruchu, bez trybu demo i bez benchmarku.

Klient wskazuje adres iPada jako bazowy URL API:

```
http://<IP_IPADA>:8080/v1
```

## Zmierzona wydajnosc

> Do wypelnienia po przebiegu na urzadzeniu. `scripts/bench.sh` generuje te tabele gotowa
> do wklejenia.

iPad Pro M5, `gemma-3-4b-it` Q4_K_M + mmproj f16, `n_ctx` 32768, KV f16, iSWA on,
pan & scan off, cache prefiksu off:

| prompt | prefill tok/s | generacja tok/s | szczyt footprint |
|---|---|---|---|
| ~1 000 tok. | _TBD_ | _TBD_ | _TBD_ |
| **~6 000 tok.** | **_TBD_** | **_TBD_** | **_TBD_** |
| ~16 000 tok. | _TBD_ | _TBD_ | _TBD_ |
| ~32 000 tok. | _TBD_ | _TBD_ | _TBD_ |

Definicje: prefill = `prompt_tokens / czas_do_pierwszego_tokenu` (wliczajac tokenizacje
i kodowanie obrazu), generacja = `completion_tokens / (czas_calkowity - czas_prefillu)`.

## Instalacja

1. Pobierz `LocalLLMServer.ipa` z artefaktow builda (zakladka Actions) albo z Releases.
2. Zainstaluj przez **AltStore** lub **Sideloadly** na darmowym Apple ID.
3. Uruchom aplikacje, zakladka **Models** -> pobierz `gemma-3-4b-it Q4_K_M + mmproj`
   (3,34 GB, pobieranie wznawialne) albo wgraj pliki `.gguf` przez Files.
4. **Load** przy modelu, potem zakladka **Server** -> **Start server**.
5. Przy pierwszym uruchomieniu iOS zapyta o zgode na siec lokalna. Bez niej serwer nie
   przyjmie zadnego polaczenia i nie powie dlaczego.

**Podpis darmowym Apple ID wygasa po 7 dniach** — potem trzeba odswiezyc aplikacje
w AltStore. Zaplanuj okno testowe.

IPA jest **niepodpisana** (brak platnego konta Apple Developer), dlatego pipeline nie
potrzebuje zadnych sekretow. Podpis nadaje narzedzie instalujace.

## Model

llama.cpp wymaga **dwoch** plikow, inaczej niz runtime, ktory trzyma model scalony:

```
ggml-org/gemma-3-4b-it-GGUF
  gemma-3-4b-it-Q4_K_M.gguf   2,49 GB   <- wagi tekstowe
  mmproj-model-f16.gguf       851 MB    <- wieza wizyjna; bez niej zadania z obrazem sa odrzucane
```

Aplikacja paruje model z projektorem po nazwie pliku i pokazuje w UI, czy para jest
kompletna. Model bez projektora obsluzy tekst, a zadanie wizyjne odrzuci bledem — zamiast
po cichu odpowiedziec z samego tekstu, co wygladaloby jak spadek jakosci.

Modele leza w katalogu Documents aplikacji (`UIFileSharingEnabled`), nie w binarce.

## Endpointy

| endpoint | rola |
|---|---|
| `POST /v1/chat/completions` | glowny: tekst, obraz w data URI, `response_format` z wymuszonym schematem |
| `GET /v1/models` | lista wczytanych modeli |
| `GET /health` | 200 gdy model wczytany, 503 gdy nie |
| `GET /v1/stats` | kolejka, licznik zadan, pamiec, termika, uptime |
| `GET /v1/selftest` | autodiagnostyka na urzadzeniu (patrz nizej) |
| `GET /logs`, `GET /logs.jsonl` | log zadan do odczytu zdalnego |

### `GET /v1/selftest`

Jedno zadanie, jeden JSON do odeslania. Sprawdza rzeczy, ktorych nie da sie zweryfikowac
bez urzadzenia, a ktore po cichu unieważniaja pomiar:

- **szablon czatu** — zwraca wyrenderowany prompt; brak `<start_of_turn>` oznacza, ze GGUF
  niesie inny szablon niz runtime odniesienia, co da inne wyjscie przy tych samych wagach;
- **tokeny obrazu** — ma byc **256** na obraz; wielokrotnosc oznacza wlaczony pan & scan
  i liczby nieporownywalne z referencja;
- **gramatyka** — kompiluje schematy wbudowane oraz **wszystkie pliki `.json` wgrane do
  `Documents/selftest/`**. Tak sprawdza sie produkcyjne schematy bez umieszczania ich w repo;
- **generacja pod gramatyka** — krotkie wyjscie i informacja, czy parsuje sie jako JSON.

## Ograniczenia, ktorych nie da sie obejsc

- **Serwer dziala tylko z aplikacja na pierwszym planie i wlaczonym ekranem.** iOS usypia
  aplikacje w tle. Aplikacja trzyma `isIdleTimerDisabled`, wiec ekran nie zgasnie, ale iPad
  musi lezec odblokowany z aplikacja na wierzchu przez caly przebieg.
- **`com.apple.developer.kernel.increased-memory-limit` jest wpisane, ale przy darmowym
  Apple ID nie zadziala** — narzedzie podpisujace je usunie. Dla modelu 4B (szczyt ~4,6 GB)
  na iPadzie 12/16 GB wystarcza bez niego. **Przy wiekszych modelach to bedzie blokada.**
- **iPady 8 GB** wywracaja sie przy dlugich promptach z modelami tej wielkosci. Brak pamieci
  konczy sie komunikatem i zwolnieniem modelu, nie wysypem: przy krytycznej presji pamieci
  model jest wyladowywany, a zadania dostaja 503, ktore klient ponawia.
- **Brak strumieniowania** — swiadomie.
- Zadania sa obslugiwane **pojedynczo, w kolejce FIFO**. Rownoleglosc zmienialaby numeryke
  i psula powtarzalnosc; polaczenia czekaja, nigdy nie sa odrzucane.

## Kryteria odbioru

Skrypty uruchamia sie **z laptopa w tej samej sieci**, nie z iPada — chodzi o udowodnienie,
ze endpoint dziala dla innego urzadzenia. Wymagaja `curl` i `jq`.

```bash
export BASE_URL=http://<IP_IPADA>:8080/v1
scripts/acceptance/run-all.sh sciezka/do/menu.jpg
```

| skrypt | co sprawdza |
|---|---|
| `01_text.sh` | poprawna odpowiedz OpenAI z niezerowym `usage` |
| `02_vision.sh` | obraz w data URI zostaje opisany; kontrola kosztu tokenowego obrazu |
| `03_schema.sh` | **zlosliwie**: prompt kaze zlamac `enum`, `required` i `minItems`; gramatyka ma na to nie pozwolic |
| `04_parallel.sh` | 20 zadan naraz, wszystkie obsluzone, zadne polaczenie nie zerwane |
| `05_long_request.sh` | zadanie ponad 10 minut konczy sie powodzeniem |
| `06_logs.sh` | log i statystyki nios tokeny, czasy i tok/s |

Tabela pomiarow do README: `scripts/bench.sh | tee bench.md` (raz z wylaczonym i raz
z wlaczonym cache'em prefiksu — patrz uwaga w skrypcie).

## Budowanie

Build robi GitHub Actions na runnerze macOS (`.github/workflows/ci.yml`): XCFramework
llama.cpp z cache'em, testy jednostkowe na symulatorze, niepodpisana IPA jako artefakt,
Release przy tagu `v*`.

Lokalnie, na macOS:

```bash
scripts/build-llama-xcframework.sh   # ~30 min za pierwszym razem
xcodegen generate
open LocalLLMServer.xcodeproj
```

Wersja llama.cpp jest przypieta w [LLAMA_CPP_TAG](LLAMA_CPP_TAG). Uzasadnienia decyzji
technicznych i parametry porownywalnosci pomiaru: [docs/DECISIONS.md](docs/DECISIONS.md).

## Co testuje CI, a czego nie moze

CI sprawdza to, co da sie sprawdzic bez urzadzenia i bez modelu: parser HTTP, kolejke
(dwadziescia rownoleglych zadan obsluzonych pojedynczo i w kolejnosci), kontrakt drutowy
oraz **konwersje JSON Schema na gramatyke GBNF**.

Czego nie sprawdzi: czy gramatyka **odrzuca** konkretny ciag — to wymaga slownika z pliku
modelu, ktorego CI nie ma. Ta czesc dzieje sie na urzadzeniu, przez `/v1/selftest`
i `03_schema.sh`.
