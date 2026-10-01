# Fixtures

## Postac na drucie

Schematy sa w postaci, w jakiej realnie przychodza po HTTP, a nie w tej, ktora generuje
framework po stronie klienta. Biblioteka kliencka przeksztalca je przed wyslaniem:

- usuwa `title`
- dodaje `"additionalProperties": false` do kazdego obiektu

Zweryfikowane na przechwyconym zadaniu. Konwerter przetestowany na surowym wyjsciu
frameworka przejdzie te testy i wywali sie na pierwszym prawdziwym zadaniu, bo dostanie
klucz, ktorego nigdy nie widzial.

## PULAPKA: klucz `_why` w wektorach odrzucenia

Wektory w `*.reject.jsonl` niosa pole `_why` z opisem, czego dany przypadek dotyczy.
Odkad schematy maja `additionalProperties: false`, **to pole samo w sobie lamie schemat**.

Konsekwencja: harness MUSI usunac `_why` przed walidacja. Inaczej kazdy wektor odrzucenia
zostanie odrzucony — ale z powodu adnotacji, a nie z powodu, ktory mial testowac. Suita
swiecilaby na zielono, nie sprawdzajac niczego.

Test tego testu: usun z jednego wektora wlasciwe naruszenie (np. wstaw poprawna wartosc
enum) i zostaw `_why`. Jesli taki wektor nadal jest odrzucany, harness nie usuwa `_why`.

## Pliki

| plik | rola |
|---|---|
| `schema_complex.json` | najtrudniejszy: `$defs`, `$ref`, `anyOf`, `enum`, `minItems`, `maxLength`, `minimum`, `additionalProperties`, glebokosc 8 |
| `schema_complex.accept.jsonl` | wyjscia, ktore gramatyka MUSI dopuscic |
| `schema_complex.reject.jsonl` | wyjscia, ktorych gramatyka NIE MOZE dopuscic |
| `schema_simple.json` | minimalny: `$defs` + `$ref` + `anyOf` |

Testuj konwerter najpierw na `schema_complex.json`. Jesli ten da poprawna gramatyke,
reszta pojdzie. Jesli nie — reszty nie warto probowac.

## Co asertowac

Poprawnosc strukturalna gramatyki nie wystarczy. Asercje maja dotyczyc wygenerowanego
wyjscia:

1. pole enumeryczne nie jest w stanie wyemitowac wartosci spoza listy
2. pole `required` nie jest w stanie zostac pominiete
3. tablica z `minItems: 1` nie jest w stanie byc pusta
4. pole wymagane-ale-nullowalne emituje wartosc albo jawny `null`, nigdy nic
5. wyjscie to goly JSON — bez ogrodzen i preambuly
6. obiekt z `additionalProperties: false` nie przyjmuje dodatkowego klucza
7. schemat nieprzekladalny daje 400, nigdy cichy fallback

Punkt 4 jest latwy do przeoczenia. Pole nullowalne, ktore NIE jest w `required`, bywa
pomijane w calosci przez mniejsze modele — a `minItems` na nieobecnym kluczu nigdy sie
nie uruchamia, wiec ograniczenie po cichu nic nie robi.
