# Szczegolowy opis `analysis.ipynb`

## Cel notebooka

- Notebook sluzy do wstepnej analizy zbioru danych z nagraniami audio.
- Jego glownym zadaniem jest sprawdzenie, jak wyglada struktura danych przed rozpoczeciem trenowania modeli.
- Analiza skupia sie na liczbie nagran w poszczegolnych klasach oraz na tym, ile unikalnych zrodel nagran wystepuje w kazdej klasie.
- Wyniki notebooka pomagaja zdecydowac, jak traktowac klasy docelowe i klasy spoza glownego zestawu komend.

## 1. Przygotowanie narzedzi

- Notebook rozpoczyna sie od zaladowania podstawowych bibliotek potrzebnych do analizy danych.
- Wykorzystywane sa narzedzia do:
  - przechodzenia po katalogach z plikami,
  - tworzenia tabel z wynikami,
  - rysowania wykresow.
- Na tym etapie notebook nie przetwarza jeszcze audio, tylko przygotowuje srodowisko do analizy struktury plikow.

## 2. Odczytanie struktury katalogow

- Notebook przechodzi przez katalog z danymi audio.
- Dla kazdego podkatalogu zapamietuje liste plikow, ktore sie w nim znajduja.
- Kazdy podkatalog odpowiada jednej klasie nagran, na przyklad konkretnej komendzie glosowej.
- Dzieki temu notebook buduje ogolny obraz tego, jakie klasy sa obecne w zbiorze danych.

## 3. Pominiecie katalogow technicznych

- Podczas analizy pomijany jest glowny katalog danych, poniewaz sam nie reprezentuje klasy.
- Pomijany jest tez katalog z szumem tla.
- Szum tla nie jest analizowany razem z komendami, poniewaz pelni inna role niz standardowe klasy slow.
- Takie pominiecie pozwala skupic pierwsza analize na klasach odpowiadajacych nagranym slowom.

## 4. Policzenie liczby nagran w kazdej klasie

- Dla kazdej klasy notebook liczy calkowita liczbe plikow audio.
- Ta wartosc pokazuje, ile probek jest dostepnych dla danej komendy lub slowa.
- Liczba probek jest wazna, poniewaz klasy z bardzo mala liczba nagran moga byc trudniejsze do nauczenia przez model.
- Klasy z bardzo duza liczba nagran moga natomiast zdominowac proces trenowania, jesli dane nie zostana odpowiednio przygotowane.

## 5. Policzenie liczby unikalnych nagran lub mowcow

- Notebook liczy takze liczbe unikalnych identyfikatorow w kazdej klasie.
- Identyfikator jest wyciagany z nazwy pliku.
- Pozwala to oszacowac, czy wiele plikow pochodzi od roznych zrodel, czy raczej od ograniczonej liczby mowcow.
- Ta informacja jest istotna, bo model powinien uczyc sie rozpoznawac komendy, a nie zapamietywac konkretne glosy.

## 6. Utworzenie tabeli podsumowujacej klasy

- Z policzonych wartosci tworzona jest tabela.
- Kazdy wiersz tabeli odpowiada jednej klasie.
- Tabela zawiera:
  - nazwe klasy,
  - liczbe wszystkich nagran,
  - liczbe unikalnych identyfikatorow.
- Klasy sa uporzadkowane alfabetycznie, zeby latwiej bylo je przegladac i porownywac.

## 7. Policzenie lacznej liczby nagran

- Notebook sumuje liczbe plikow ze wszystkich analizowanych klas.
- Wynik pokazuje calkowity rozmiar glownej czesci zbioru danych.
- Jest to szybka informacja o skali problemu i ilosci danych dostepnych do dalszych etapow projektu.

## 8. Wizualizacja rozkladu wszystkich klas

- Notebook tworzy wykres slupkowy dla wszystkich analizowanych klas.
- Wykres pokazuje jednoczesnie:
  - liczbe wszystkich nagran w klasie,
  - liczbe unikalnych identyfikatorow w klasie.
- Wizualizacja pozwala szybko zobaczyc, ktore klasy maja najwiecej danych, a ktore sa mniej liczne.
- Pomaga tez ocenic, czy rozklad danych jest rownomierny.

## 9. Wybor klas docelowych

- Notebook definiuje zestaw klas, ktore sa glowne dla zadania klasyfikacji komend glosowych.
- Sa to komendy:
  - `yes`,
  - `no`,
  - `up`,
  - `down`,
  - `left`,
  - `right`,
  - `on`,
  - `off`,
  - `stop`,
  - `go`.
- Te klasy stanowia podstawowy zestaw slow, ktore model ma rozpoznawac osobno.

## 10. Utworzenie klasy `unknown`

- Wszystkie klasy, ktore nie naleza do zestawu klas docelowych, sa laczone w jedna wspolna kategorie `unknown`.
- Oznacza to, ze model nie musi odrozniac kazdego slowa spoza glownego zestawu.
- Zamiast tego ma nauczyc sie, ze takie nagranie nie jest jedna z docelowych komend.
- Takie podejscie upraszcza problem klasyfikacji i lepiej odpowiada celowi projektu.

## 11. Przygotowanie tabeli dla klas uzywanych w projekcie

- Notebook tworzy nowa tabele zawierajaca tylko klasy docelowe oraz klase `unknown`.
- Dla klasy `unknown` sumowane sa wartosci ze wszystkich klas spoza glownego zestawu.
- Tabela pokazuje, jak bedzie wygladal rozklad danych po uproszczeniu etykiet.
- Jest to wazne, bo ten rozklad jest blizszy temu, co pozniej zobaczy model podczas uczenia.

## 12. Wizualizacja klas docelowych i `unknown`

- Notebook tworzy drugi wykres slupkowy.
- Tym razem wykres obejmuje tylko klasy docelowe oraz klase `unknown`.
- Pozwala to porownac liczebnosc glownych komend z laczna liczba pozostalych slow.
- Wykres pomaga ocenic, czy kategoria `unknown` jest znacznie wieksza od pojedynczych klas docelowych.
- Taka informacja moze wskazywac, ze podczas trenowania trzeba bedzie uwazac na niezbalansowanie klas.

## Znaczenie notebooka dla calego projektu

- `analysis.ipynb` jest pierwszym krokiem projektu.
- Nie trenuje modeli i nie modyfikuje danych audio.
- Dostarcza podstawowej wiedzy o strukturze zbioru danych.
- Pokazuje, jakie klasy sa dostepne i jak licznie sa reprezentowane.
- Uzasadnia pozniejsze traktowanie klas spoza glownego zestawu jako `unknown`.
- Pomaga przygotowac decyzje wykorzystywane w kolejnych notebookach, szczegolnie podczas trenowania modeli.
