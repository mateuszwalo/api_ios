# Decyzje techniczne i parametry runtime

Co jest ustalone, dlaczego, i czego nie wolno ruszyc bez swiadomej decyzji.

Aplikacja powstala, zeby zmierzyc wydajnosc `gemma-3-4b-it` Q4_K_M na Apple Silicon
na realnym ruchu OpenAI-owym. Wartosci w tabeli 1 odwzorowuja konfiguracje odniesienia
(ta sama praca pod ollama) — zmiana ktorejkolwiek uniewaznia porownanie wynikow.

## 1. Parametry, ktore musza sie zgadzac z konfiguracja odniesienia

| parametr | wartosc | uzasadnienie |
|---|---|---|
| `n_ctx` | 32768 | jak w Modelfile odniesienia (nie domyslne ollamy 4096) |
| `temperature` | 0 | nadpisywalne przez zadanie, reszta samplingu nie |
| `top_k` | 1 | na sztywno |
| `top_p` | 1 | na sztywno |
| kwantyzacja | Q4_K_M | `ggml-org/gemma-3-4b-it-GGUF` |
| mmproj | f16 (851 MB) | to samo repo |
| KV cache | f16 | domyslne ollamy |
| iSWA | wlaczone | Gemma 3: pelne okno dostaje 5 z 34 warstw |
| flash attention | on | domyslne na Metalu |
| mmap | on | mniejszy szczyt pamieci |
| pan & scan | WYLACZONY | jeden kafelek = 256 tokenow na obraz |
| szablon czatu | Gemma 3, stop `<end_of_turn>` | czytany z metadanych GGUF, nie z naszego kodu |
| cache prefiksu KV | domyslnie WYLACZONY | czysty prefill w pomiarze; osobny przebieg z wlaczonym |

Kontrola poprawnosci kafelkowania jest tania: krotki prompt + jeden obraz musi dac
`prompt_tokens` rzedu 298 (256 obrazu + ~42 tekstu). Wartosc rzedu 550 czy 810 oznacza
wlaczony pan & scan i natychmiast psuje porownywalnosc.

## 2. Decyzje architektoniczne

**Wlasny serwer Swift (Hummingbird 2) + cienki most do llama.cpp.**
Nie linkujemy `tools/server` llama.cpp jako calosci, mimo ze ma gotowa zgodnosc z OpenAI:
w jego petli slotow nie da sie wylaczyc reuse prefiksu KV (slot zawsze doklada do
wspolnego prefiksu poprzedniego zadania), a domyslnie wymagamy czystego prefillu.
Wlasna petla daje tez `llama_memory_clear()` miedzy zadaniami, per-request logi do UI,
anulowanie i pomiar footprintu dokladnie w fazie generacji.

**Ponownie uzywamy trzech klockow llama.cpp zamiast pisac je od nowa:**

| klocek | skad | dlaczego nie sami |
|---|---|---|
| `json_schema_to_grammar` | `common/` | obsluguje `$defs`/`$ref`/`anyOf`/`enum`/`minItems`/`maxLength`/`minimum` |
| szablon czatu | `llama_model_chat_template` + `llama_chat_apply_template` | szablon z metadanych GGUF = zero ryzyka rozjazdu z ollamowym |
| wizja | `mtmd` | jedyna droga mmproj + Gemma 3 na iOS |

`common/` nie jest pakowany przez upstreamowy `build-xcframework.sh` — stad wlasny
skrypt (`scripts/build-llama-xcframework.sh`) z `LLAMA_BUILD_COMMON=ON`.

**Wersja llama.cpp przypieta** w `LLAMA_CPP_TAG`. Zachowanie `json_schema_to_grammar`
i obsluga wizji gemma3 zmienialy sie miedzy wersjami; powtarzalnosc pomiaru wymaga pinu.

**llama.cpp nie jest submodulem** — skrypt klonuje przypiety tag shallow. Repo zostaje
male, a klucz cache'u CI to wprost zawartosc `LLAMA_CPP_TAG`.

**`GGML_METAL_EMBED_LIBRARY=ON`** — shadery Metal w binarce. Kilkaset kB wiecej
i kilkaset ms dluzszy pierwszy start, w zamian znika klasa bledow "dziala w symulatorze,
nie dziala na urzadzeniu" przy IPA rozprowadzanej sideloadem.

**iOS 17.0**, XcodeGen (`project.yml`), jeden slot modelu, kolejka FIFO bez limitu.

**Serwer HTTP wlasny, na Network.framework — nie Hummingbird.** Pierwotnie planowalismy
Hummingbirda 2. Zmiana po przyjrzeniu sie wymaganiom: potrzebujemy braku jakiegokolwiek
timeoutu (odpowiedz moze isc 20 minut), wlaczonego `SO_KEEPALIVE` z krotkim interwalem
(cisza w gniezdzie przez 20 minut to dokladnie to, co czysci tablica NAT w routerze) oraz
wykrycia rozlaczenia klienta W TRAKCIE liczenia. Ustawienie tych trzech rzeczy w cudzym
frameworku wymaga znalezienia i nadpisania jego domyslnych wartosci; napisanie ich wprost
to ~400 linii, w tym czysty parser HTTP pokryty testami. Przy okazji znika zaleznosc
SwiftPM z CI i cala klasa niespodzianek przy aktualizacji wersji. Protokol jest prosty:
jedno POST z JSON-em naraz, bez strumieniowania.

**Gramatyka testowana dwustopniowo.** Konwersja schema -> GBNF jest czysta i nie potrzebuje
modelu, wiec idzie w CI na symulatorze. Natomiast rozstrzygniecie, czy gramatyka AKCEPTUJE
dany ciag, wymaga slownika, czyli pliku modelu, ktorego CI nie ma. Dlatego akceptacja
i odrzucenie sa sprawdzane na urzadzeniu przez `GET /v1/selftest` — i na prawdziwych
schematach, ktore wgrywa sie do `Documents/selftest/` przez Files, nigdy do repo.

## 3. Decyzje kontraktu API

- `model` w zadaniu ignorowany przy wyborze, odbijany w odpowiedzi.
- **Limit generacji przychodzi jako `max_completion_tokens`, nie `max_tokens`.**
  Zweryfikowane na przechwyconym zadaniu: klient wysyla to pierwsze i POMIJA drugie.
  Czytamy oba, `max_completion_tokens` ma pierwszenstwo. Serwer czytajacy tylko
  `max_tokens` generowalby bez limitu az do sciany kontekstu, i to po cichu.
- `temperature`, `stop`, `seed` respektowane; reszta cicho ignorowana. W praktyce
  przychodzi tylko `temperature` — pelne cialo zadania to szesc pol:
  `max_completion_tokens`, `messages`, `model`, `response_format`, `stream` (zawsze
  `false`), `temperature`.
- **W tresci wiadomosci obraz idzie PRZED tekstem.** Obslugujemy obie kolejnosci,
  ale taka przychodzi realnie.
- `prompt + limit > n_ctx` -> przyciecie limitu, `finish_reason: "length"`.
  400 tylko gdy sam prompt sie nie miesci w oknie.
- **`json_schema` niesie wylacznie `name` i `schema` — klucza `strict` NIE MA.**
  Nie wymagamy go i nie traktujemy braku jako przyzwolenia na lagodniejsze egzekwowanie.
- **Schemat na drucie jest juz przeksztalcony przez biblioteke kliencka**: bez `title`,
  za to z `"additionalProperties": false` na kazdym obiekcie. Konwerter musi ten klucz
  przyjac, a nie odrzucic jako nieobslugiwany.
- W odpowiedzi `id`, `object` (doslownie `"chat.completion"`), `created`, `model`
  i `choices` sa WYMAGANE przez model odpowiedzi SDK — brak ktoregokolwiek wywala
  parsowanie u klienta, zanim ktokolwiek spojrzy na tresc.
- Model niewczytany -> 503 + `Retry-After`. Klient OpenAI ponawia 429/500/502/503/504/529;
  400 zabiloby caly przebieg.
- Schemat nieprzekladalny na gramatyke -> 400 z czytelnym bledem. NIGDY fallback na
  podpowiedz w prompcie: cicha degradacja skazilaby pomiar (gorszy wynik przypisalibysmy
  sprzetowi albo modelowi).
- `timings` (prefill_ms, decode_ms, prefill_tps, decode_tps) obok `usage`, nie wewnatrz —
  ksztalt `usage` jest kontraktem.
- Bez strumieniowania. `SO_KEEPALIVE` wlaczony, zero idle timeoutow (zadania trwaja minuty,
  klient ma timeout rzedu godzin).
- Zerwane polaczenie klienta -> anulowanie zadania, zeby nie blokowalo kolejki.

## 4. Do rozstrzygniecia w testach na urzadzeniu

- Schemat bez `required` na poziomie root daje gramatyke, w ktorej model MOZE zwrocic
  `{}`. Odwzorowujemy schemat wiernie; jesli w praktyce wyjda puste odpowiedzi,
  dokladamy przelacznik "traktuj wszystkie wlasciwosci jako wymagane". Uwaga: nie mozemy
  tego uzaleznic od `strict`, bo klient tego klucza nie wysyla (patrz sekcja 3) —
  musialby to byc przelacznik w UI, nie odczyt z zadania.
- `n_batch` / `n_ubatch` / liczba watkow — dobrac eksperymentalnie, zaraportowac
  w README, wystawic w UI.
