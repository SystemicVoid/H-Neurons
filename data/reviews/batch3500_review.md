# Batch 3500 Review

Reviewed file: `data/gemma3_4b_TriviaQA_consistency_samples.jsonl`  
Batch size: 3,500 questions x 10 responses each = 35,000 judged responses

## 1. Basic Counts

| Metric | Count |
| --- | --- |
| Total entries | 3500 |
| All-correct | 1435 |
| All-incorrect | 1680 |
| Mixed | 385 |
| Uncertain-containing | 0 |
| All-correct + all-incorrect + mixed | 3500 |

## 2. Substring Loophole Quantification

| Metric | Count |
| --- | --- |
| All-correct entries | 1435 |
| All-correct entries with a >3-word substring-match response | 96 |

| QID | Matched alias | Words | Example response |
| --- | --- | --- | --- |
| qz_50 | clyde barrow | 6 | "Bonnie Parker: Barrow\\n Clyde Barrow: Barrow" |
| qz_52 | orville wright | 4 | "Wilbur and Orville Wright" |
| qz_59 | Home Alone 2 | 7 | "Home Alone 2: Lost in New York" |
| tc_115 | and that s way it is | 6 | "And that’s the way it is." |
| tc_138 | Henry james | 4 | "Henry James\\nJuan Gris" |
| tc_247 | The Sound Of Silence | 8 | "\\"The Sound of Silence\\" by Simon & Garfunkel" |
| qz_286 | Three wise men | 6 | "The Three Wise Men (or Magi)" |
| tc_303 | White | 4 | "Stormtrooper armor was white." |
| tc_331 | Liberty Enlightening the World | 4 | "Liberty Enlightening the World" |
| qz_358 | The Godfather : Part II | 4 | "The Godfather Part II" |
| qz_400 | she was playing monopoly | 4 | "She was playing Monopoly." |
| tc_434 | Three days and three nights | 5 | "Three days and three nights" |
| qz_448 | Gone with the Win | 4 | "Gone with the Wind" |
| qz_670 | ear | 6 | "Cochlea: Inner ear; Stirrup: Middle ear" |
| qz_753 | To Kill a Mockingbird | 4 | "To Kill a Mockingbird" |
| tc_851 | art historian | 5 | "Art historian and art advisor" |
| qz_854 | Cassius Marcellus Clay Jr | 4 | "Cassius Marcellus Clay Jr." |
| tc_863 | long walk to freedom | 4 | "Long Walk to Freedom" |
| qz_956 | submarine | 8 | "A Soviet submarine equipped with a stealth technology." |
| qz_959 | International Man of Mystery | 6 | "Austin Powers: International Man of Mystery" |
| qz_1092 | Santa Cruz | 4 | "Santa Cruz de Tenerife" |
| tc_1122 | A Space Odyssey | 4 | "2001: A Space Odyssey" |
| tc_1126 | Judgment Day | 4 | "Terminator 2: Judgment Day" |
| tc_1140 | Empire strikes back | 4 | "The Empire Strikes Back" |
| tc_1151 | Silence of the lambs | 5 | "The Silence of the Lambs" |
| tc_1170 | Kevin Costner | 4 | "Kevin Costner, Kevin Costner" |
| qz_1174 | last christmas by wham | 4 | "Last Christmas by Wham" |
| qz_1179 | The web | 4 | "The World Wide Web" |
| tc_1204 | kit kat | 4 | "The Kit Kat Club" |
| qz_1305 | The Sword In the Stone | 5 | "The Sword in the Stone" |
| qz_1307 | angel of north | 4 | "Angel of the North" |
| qz_1348 | Genesis, Exodus, Leviticus, Numbers | 4 | "Genesis, Exodus, Leviticus, Numbers" |
| tc_1361 | Baby Doc | 16 | "\\"Baby Doc\\" Duvalier was known for his lavish lifestyle, patronage of the arts, and perceived corruption." |
| tc_1374 | VTOL aircraft | 6 | "Vertical takeoff and landing (VTOL) aircraft" |
| tc_1376 | oil | 5 | "Crude oil and refined products" |
| qz_1383 | luther king jr | 4 | "Martin Luther King Jr." |
| tc_1389 | Blood On The Tracks | 4 | "Blood on the Tracks" |
| tc_1445 | terry bradshaw | 6 | "1979: Terry Bradshaw\\n1980: Earl Campbell" |
| tc_1534 | Super Bowl | 6 | "National Football League (NFL) Super Bowl" |
| qz_1546 | Ben-Hur, Titanic and The Lord of the Rings: The Return of the King | 13 | "Ben-Hur, Titanic, and The Lord of the Rings: The Return of the King" |
| tc_1551 | America's cup | 16 | "Australia II won the America's Cup in 1983. America 3 won the America's Cup in 1974." |
| tc_1598 | Gulf of Tonkin incident | 4 | "Gulf of Tonkin Incident" |
| qz_1600 | Frankly, my dear, I don't give a damn | 8 | "“Frankly, my dear, I don’t give a damn.”" |
| tc_1643 | Gorbachev | 13 | "Margaret Thatcher described Mikhail Gorbachev as “a man we can do business with.”" |
| tc_1644 | Dalai-lama | 4 | "The 14th Dalai Lama" |
| tc_1810 | Le Guin | 4 | "Ursula K. Le Guin" |
| tc_1937 | priest | 4 | "Former priest and politician" |
| tc_1957 | ed sullivan | 4 | "The Ed Sullivan Show" |
| tc_1958 | Peres | 7 | "Yasser Arafat, Shimon Peres, and Yitzhak Rabin" |
| tc_1988 | annie get your gun | 4 | "Annie Get Your Gun" |
| tc_2053 | Shannon | 5 | "River Shannon, Shannon hydroelectric power" |
| qz_2091 | british overseas airways corporation | 4 | "British Overseas Airways Corporation" |
| tc_2113 | Alfred Stieglitz | 6 | "Alfred Stieglitz and Lothrup W. Whitshed" |
| tc_2130 | NEW YORK | 5 | "New York City, New York" |
| tc_2190 | Ain't No Mountain High Enough | 5 | "\\"Ain't No Mountain High Enough\\"" |
| qz_2215 | one flew over cuckoo s nest | 6 | "One Flew Over the Cuckoo's Nest" |
| tc_2251 | cds | 4 | "CDs and digital downloads" |
| tc_2264 | luther king jr | 4 | "Martin Luther King Jr." |
| tc_2272 | detroit mich | 8 | "Chrysler: Auburn Hills, Michigan\\nGeneral Motors: Detroit, Michigan" |
| tc_2288 | san francisco ballet | 4 | "San Francisco Ballet School" |
| tc_2310 | balanchine | 5 | "Isadora Duncan and George Balanchine" |
| qz_2413 | Karl marx | 5 | "Karl Marx and Friedrich Engels" |
| qz_2447 | How I Learned To Stop Worrying and Love The Bomb | 13 | "Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb" |
| qz_2462 | The seven dwarfs | 6 | "Snow White and the Seven Dwarfs" |
| qz_2469 | A lighthouse | 4 | "The Lighthouse of Alexandria" |
| tc_2560 | scutari | 4 | "Scutari (present-day Istanbul, Turkey)" |
| tc_2561 | Jacobite rising | 5 | "The Jacobite Rising of 1746" |
| qz_2566 | bonnie elizabeth parker | 7 | "Bonnie Elizabeth Parker and Clyde Frederick Thompson" |
| tc_2572 | The Commonwealth | 4 | "The Commonwealth of England" |
| tc_2585 | War of Spanish Succession | 5 | "War of the Spanish Succession" |
| tc_2589 | Great Leap | 4 | "The Great Leap Forward" |
| qz_2614 | Some Like it Hot | 4 | "Some Like It Hot" |
| tc_2631 | The British East India Company | 5 | "The British East India Company" |
| tc_2632 | Drug-trafficking | 7 | "Drug trafficking and conspiracy to traffic narcotics." |
| tc_2640 | The NSDAP | 6 | "National Socialist German Workers' Party (NSDAP)" |
| tc_2685 | Little Bighorn | 5 | "Battle of the Little Bighorn" |
| tc_2688 | dutc | 19 | "Dutch-Indonesian (specifically, her parents were of Indonesian and European descent, and she was born in Dutch Indesia, now Indonesia)." |
| qz_2816 | Buffy: The Vampire Slayer | 4 | "Buffy the Vampire Slayer" |
| tc_2817 | Kangaro | 10 | "Wallabies, kangaroos, wombats, bindi, echidnas, cassowaries, rusa deer (New Guinea)." |
| tc_2831 | krill | 8 | "Krill, small fish, and other small marine animals." |
| qz_2876 | Springfield Nuclear | 4 | "Springfield Nuclear Power Plant" |
| tc_2879 | During sleep | 17 | "During sleep, primarily in the later stages of NREM sleep and during the transition between sleep stages." |
| tc_2884 | Jugular | 7 | "Internal jugular vein and external jugular vein" |
| tc_2902 | The Brain | 6 | "Cerebellum: Brain\\nMedulla: Brainstem\\nHypothalamus: Brain" |
| tc_2917 | Foot | 4 | "Toes and foot arches" |
| tc_2920 | Hormones | 22 | "Insulin, glucagon, thyroid hormones (T3, T4), cortisol, adrenaline (epinephrine), estrogen, progesterone, testosterone, growth hormone, melatonin, oxytocin, antidiuretic hormone (ADH), parathyroid hormone (PTH)." |
| tc_2945 | pancrea | 8 | "Islets of Langerhans are found in the pancreas." |
| tc_2953 | Endorphins | 5 | "Endorphins, enkephalins, dynorphins, oxytocin, vasopressin." |
| tc_2960 | Neck | 4 | "Sides of the neck" |
| tc_2980 | Mohs' scale | 10 | "Mohs scale, Vickers hardness, Rockwell hardness, Knoop hardness, Brinell hardness." |
| tc_3038 | The trade in endangered species | 4 | "Trade in endangered species" |
| tc_3042 | Nitrogen oxides | 4 | "Nitrogen oxides, sulfur dioxide." |
| tc_3103 | white dwarf | 7 | "A red giant, then a white dwarf." |
| tc_3118 | Nited States | 4 | "United States of America" |
| tc_3132 | light years | 7 | "Light-years, parsecs, astronomical units (AU), kiloparsecs, megaparsecs." |
| tc_3135 | Expansion of the Universe | 5 | "The expansion of the universe." |

## 3. Response Length Distribution

| Percentile | Word count |
| --- | --- |
| p5 | 1 |
| p10 | 1 |
| p25 | 1 |
| p50 | 2 |
| p75 | 2 |
| p90 | 3 |
| p95 | 4 |
| p99 | 8 |
| max | 38 |

| QID | Words | Response |
| --- | --- | --- |
| qz_71 | 18 | "Copernicus argued the Sun was at the center.\\n\\nThe common belief was the Earth was at the center." |
| qz_71 | 18 | "Copernicus argued the Sun was at the center.\\n\\nThe common belief was the Earth was at the center." |
| qz_71 | 18 | "Copernicus argued the Sun was at the center.\\n\\nThe common belief was the Earth was at the center." |
| qz_71 | 27 | "Copernicus argued the Sun; the Earth and other planets revolved around the Sun.\\n\\nThe common belief was that the Earth was at the center of the universe." |
| tc_1233 | 16 | "“The rain patters on the windowpane,\\nThe rain patters on the pane.\\nAnd I’m happy again.”" |
| tc_1233 | 21 | "“The rain patters on the windowpane,\\nAnd I’m just a flower that’s in full bloom.\\nI’m happy now, I’m happy now.”" |
| qz_1255 | 29 | "Believe, O believe, in the bright Karish-an,\\nBelieve, O believe, in the bright Karish-an,\\nThat the dawn from Ireland will surely rise,\\nAnd spread its benison.\\n\\nBelieve, O believe," |
| qz_1255 | 35 | "Believe, O believe\\nIn yourself and in your might,\\nOh, Ireland, your sons again will strive,\\nTo build up your glory and your light.\\n\\nBelieve, O believe\\nIn yourself and in your might,\\nOh, Ireland" |
| qz_1255 | 35 | "Believe, O believe, in the light that is shining,\\nBelieve, O believe, it will guide thee aright.\\nBelieve, O believe, in the love that is beaming,\\nBelieve, O believe, and you shall find the light" |
| qz_1255 | 37 | "Believe, little soldier, in the bright morning star\\nThat the dawn of hope is coming over the sea\\n**Hearts resilient, ever brave and true,**\\n**Will guide our land to victory anew.**\\n\\n**So raise your voice and sing" |
| qz_1255 | 38 | "Believe, little one, in the bright star of hope,\\nThat shines above the earth and guides our way.\\nBelieve, little one, in the bright star of hope,\\nThat shines above the earth and guides our way.\\n\\nThe emerald" |
| tc_1361 | 16 | "\\"Baby Doc\\" Duvalier was known for his lavish lifestyle, patronage of the arts, and perceived corruption." |
| tc_1361 | 16 | "“Baby Doc” Duvalier was known for his lavish lifestyle, patronage of the arts, and alleged corruption." |
| tc_1361 | 17 | "“Baby Doc” Duvalier was known for his lavish lifestyle, corruption, and reliance on the Duvalier family dynasty." |
| tc_1361 | 17 | "“Baby Doc” Duvalier was known for his lavish lifestyle, patronage of the arts, and increasingly authoritarian rule." |
| tc_1361 | 19 | "“Baby Doc” Duvalier was known for his lavish lifestyle, patronage of the arts, and widespread corruption during his presidency." |
| tc_1361 | 21 | "“Baby Doc” Duvalier was known for his lavish lifestyle, patronage of the arts, and a system of widespread corruption and repression." |
| tc_1361 | 21 | "“Baby Doc” Duvalier was known for his lavish lifestyle, patronage of the arts, and increasingly authoritarian rule, particularly in the 1980s." |
| tc_1361 | 22 | "\\"Baby Doc\\" Duvalier was known for his lavish lifestyle, patronage of the arts, and increasingly authoritarian rule, particularly during the late 1980s." |
| tc_1361 | 23 | "“Baby Doc” – a benevolent dictator known for his lavish lifestyle, patronage of the arts, and use of social programs to maintain popularity." |
| tc_1550 | 18 | "Shinnecock Hills Golf Club hosted the US Open in 1994 and 1997.\\n\\nSeve Ballesteros (1994)\\nTiger Woods (1997)" |
| tc_1551 | 16 | "Australia II won the America's Cup in 1983. America 3 won the America's Cup in 1974." |
| tc_1551 | 16 | "Australia II won the America's Cup in 1983. America 3 won the America's Cup in 1974." |
| tc_1551 | 16 | "Australia II won the America's Cup in 1983. America 3 won the America's Cup in 1974." |
| tc_1551 | 16 | "Australia II won the America's Cup in 1983. America 3 won the America's Cup in 1977." |
| tc_1551 | 16 | "Australia II won the America's Cup in 1983; America 3 won the America's Cup in 1977." |
| tc_1551 | 16 | "Australia II won the America's Cup in 1983; America 3 won the America's Cup in 1977." |
| tc_1551 | 16 | "Australia II won the America’s Cup in 1983. America 3 won the America’s Cup in 1974." |
| tc_1551 | 16 | "Australia II won the America’s Cup in 1983; America 3 won the America’s Cup in 1974." |
| tc_1551 | 16 | "Australia II won the America’s Cup in 1983; America 3 won the America’s Cup in 1977." |
| tc_1551 | 16 | "Australia II won the America’s Cup in 1983; America 3 won the America’s Cup in 1977." |
| qz_1798 | 28 | "1. Gary Brooker\\n2. The Carpenters\\n3. David Bowie\\n4. The Rolling Stones\\n5. Elton John\\n6. Billy Joe Shaver\\n7. The Osmonds\\n8. Neil Diamond\\n9. Smokey" |
| qz_1798 | 29 | "1. Gary Wright\\n2. Jerry Butler\\n3. Paul McCartney and Wings\\n4. Robbie Dupree\\n5. Charlie Rich\\n6. Harold Faltermeyer\\n7. David Bowie\\n8. The Osmonds\\n9. Willie Nelson" |
| tc_1838 | 18 | "Through a combination of critically acclaimed, innovative films that garnered international recognition, particularly at the Cannes Film Festival." |
| tc_1838 | 19 | "Through his groundbreaking, humanist, and technically innovative films that garnered international acclaim and awards, particularly starting with *Pather Panchali*." |
| tc_1838 | 21 | "Through a combination of groundbreaking filmmaking, critical acclaim, international recognition, and a unique artistic vision that blended storytelling with technical innovation." |
| tc_1838 | 22 | "Through his groundbreaking, humanist films that garnered international acclaim and numerous awards, particularly the Palme d'Or at Cannes in 1992 for *Sérendipité*." |
| tc_1838 | 24 | "Through his groundbreaking, humanist films that garnered international acclaim and numerous awards, particularly the Palme d'Or at Cannes in 1992 for *The Apu Trilogy*." |
| tc_1838 | 26 | "Through his groundbreaking and critically acclaimed films in the 1960s and 70s, particularly *Pather Panchali*, establishing him as one of the greatest filmmakers of all time." |
| tc_1993 | 23 | "A man who was born, lived, died, and rose from the dead, and whose story is a tale of love, betrayal, and redemption." |
| qz_2008 | 20 | "He was suspended by his wrists and feet, likely to simulate death for ritualistic purposes, and slowly starved to death." |
| qz_2008 | 21 | "He was laid in a trough of brine to preserve his body, and died from the effects of the salt solution." |
| qz_2008 | 21 | "He was suspended by his wrists and feet, likely to simulate death for ritualistic purposes, and left to die of exposure." |
| qz_2008 | 21 | "He was suspended by his wrists and feet, likely to simulate death for ritualistic purposes, and left to die of exposure." |
| qz_2008 | 23 | "He was suspended by his wrists and feet, likely as part of a ritualistic practice to simulate death for a period of time." |
| tc_2404 | 18 | "A system of forced labor used by the SS to exploit prisoners of war and concentration camp inmates." |
| tc_2404 | 20 | "A system of forced labor used by the SS to exploit prisoners in concentration camps and through external labor agreements." |
| tc_2404 | 21 | "A system of forced labor used by the Nazi regime to exploit prisoners of war, concentration camp inmates, and other groups." |
| tc_2688 | 19 | "Dutch-Indonesian (specifically, her parents were of Indonesian and European descent, and she was born in Dutch Indesia, now Indonesia)." |
| tc_2759 | 16 | "Hydrogen, helium, carbon, nitrogen, oxygen, phosphorus, sulfur, chlorine, sodium, potassium, calcium, iron, copper, gold, silver, uranium." |
| tc_2759 | 16 | "Hydrogen, helium, carbon, nitrogen, oxygen, phosphorus, sulfur, chlorine, sodium, potassium, calcium, iron, copper, gold, silver, uranium." |
| tc_2759 | 16 | "Hydrogen, helium, oxygen, nitrogen, carbon, phosphorus, sulfur, chlorine, sodium, potassium, calcium, magnesium, aluminum, iron, silicon, uranium." |
| tc_2759 | 17 | "Hydrogen, helium, carbon, nitrogen, oxygen, phosphorus, sulfur, chlorine, sodium, potassium, calcium, magnesium, aluminum, silicon, iron, and copper." |
| tc_2759 | 19 | "Hydrogen, helium, carbon, nitrogen, oxygen, phosphorus, sulfur, chlorine, sodium, potassium, calcium, iron, magnesium, aluminum, silicon, argon, and many others." |
| tc_2759 | 19 | "Hydrogen, helium, carbon, nitrogen, oxygen, phosphorus, sulfur, chlorine, sodium, potassium, calcium, magnesium, aluminum, silicon, iron, copper, gold, silver, platinum." |
| tc_2817 | 16 | "Kangaroos, wallabies, wombats, bilbies, quokkas, echidnas, cassowaries, tarsiers, tree kangaroos, possums, and various introduced deer species." |
| tc_2817 | 21 | "Kangaroos, wallabies, wombats, bilbies, quokkas, cassowaries, bindi, echidnas, rusa deer, fallow deer (Tasmania), red deer (Tasmania), mountain deer (New Guinea), highland" |
| tc_2879 | 17 | "During sleep, particularly in the later stages of NREM sleep and during the transition between sleep stages." |
| tc_2879 | 17 | "During sleep, particularly in the later stages of NREM sleep and during the transition between sleep stages." |
| tc_2879 | 17 | "During sleep, particularly in the later stages of NREM sleep, and during the transition between sleep stages." |
| tc_2879 | 17 | "During sleep, particularly in the later stages of NREM sleep, and during the transition between sleep stages." |
| tc_2879 | 17 | "During sleep, particularly in the later stages of NREM sleep, and during the transition between sleep stages." |
| tc_2879 | 17 | "During sleep, primarily in the later stages of NREM sleep and during the transition between sleep stages." |
| tc_2920 | 18 | "Insulin, glucagon, thyroid hormones (T3, T4), cortisol, adrenaline (epinephrine), estrogen, progesterone, testosterone, growth hormone, melatonin, leptin, oxytocin, vasopressin." |
| tc_2920 | 20 | "Insulin, cortisol, thyroid hormones, adrenaline, estrogen, progesterone, testosterone, growth hormone, melatonin, oxytocin, erythropoietin, antidiuretic hormone, parathyroid hormone, calcitonin, gastrin, cholecystokinin" |
| tc_2920 | 22 | "Insulin, glucagon, thyroid hormones (T3, T4), cortisol, adrenaline (epinephrine), estrogen, progesterone, testosterone, growth hormone, melatonin, oxytocin, antidiuretic hormone (ADH), parathyroid hormone (PTH)." |
| tc_2920 | 22 | "Insulin, glucagon, thyroid hormones (T3, T4), cortisol, adrenaline (epinephrine), estrogen, progesterone, testosterone, growth hormone, melatonin, oxytocin, antidiuretic hormone (ADH), parathyroid hormone (PTH)." |
| tc_2920 | 22 | "Insulin, glucagon, thyroid hormones (T3, T4), cortisol, adrenaline (epinephrine), estrogen, progesterone, testosterone, growth hormone, melatonin, oxytocin, antidiuretic hormone (ADH), parathyroid hormone (PTH)." |
| tc_2920 | 22 | "Insulin, glucagon, thyroid hormones (T3, T4), cortisol, adrenaline (epinephrine), estrogen, progesterone, testosterone, growth hormone, melatonin, oxytocin, antidiuretic hormone (ADH), parathyroid hormone (PTH)." |
| tc_2920 | 22 | "Insulin, thyroid hormones (T3, T4), cortisol, adrenaline (epinephrine), estrogen, progesterone, testosterone, growth hormone, melatonin, oxytocin, antidiuretic hormone (ADH), parathyroid hormone (PTH), calcitonin." |
| tc_2920 | 22 | "Insulin, thyroid hormones (T3, T4), cortisol, adrenaline (epinephrine), estrogen, progesterone, testosterone, growth hormone, melatonin, oxytocin, antidiuretic hormone (ADH), parathyroid hormone (PTH), calcitonin." |
| tc_2920 | 22 | "Insulin, thyroid hormones (T3, T4), cortisol, adrenaline (epinephrine), estrogen, progesterone, testosterone, growth hormone, melatonin, oxytocin, antidiuretic hormone (ADH), parathyroid hormone (PTH), calcitonin." |
| tc_2920 | 22 | "Insulin, thyroid hormones (T3, T4), cortisol, adrenaline (epinephrine), estrogen, progesterone, testosterone, growth hormone, melatonin, oxytocin, antidiuretic hormone (ADH), parathyroid hormone (PTH), calcitonin." |

## 4. Multiline Response Count

| Metric | Count |
| --- | --- |
| Responses containing a newline | 113 |

| QID | Multiline responses |
| --- | --- |
| qz_50 | 10 |
| qz_71 | 4 |
| tc_138 | 10 |
| tc_171 | 10 |
| qz_670 | 8 |
| qz_864 | 1 |
| tc_1233 | 2 |
| qz_1255 | 5 |
| tc_1445 | 10 |
| tc_1476 | 2 |
| tc_1550 | 2 |
| qz_1658 | 10 |
| qz_1685 | 1 |
| tc_1733 | 10 |
| qz_1798 | 4 |
| tc_1801 | 1 |
| tc_2272 | 10 |
| qz_2477 | 1 |
| qz_2566 | 2 |
| tc_2902 | 10 |

## 5. Mixed-Label Breakdown

| QID | True count | False count | Uncertain count | Distinct normalized responses |
| --- | --- | --- | --- | --- |
| tc_3 | 4 | 6 | 0 | 6 |
| tc_15 | 6 | 4 | 0 | 3 |
| tc_18 | 2 | 8 | 0 | 9 |
| qz_26 | 9 | 1 | 0 | 2 |
| tc_26 | 7 | 3 | 0 | 2 |
| qz_30 | 9 | 1 | 0 | 2 |
| tc_30 | 3 | 7 | 0 | 3 |
| qz_45 | 7 | 3 | 0 | 2 |
| qz_53 | 4 | 6 | 0 | 2 |
| tc_77 | 8 | 2 | 0 | 2 |
| qz_82 | 5 | 5 | 0 | 2 |
| qz_87 | 8 | 2 | 0 | 4 |
| tc_97 | 1 | 9 | 0 | 6 |
| tc_100 | 1 | 9 | 0 | 5 |
| tc_103 | 6 | 4 | 0 | 2 |
| tc_113 | 4 | 6 | 0 | 3 |
| tc_139 | 3 | 7 | 0 | 5 |
| tc_145 | 1 | 9 | 0 | 5 |
| qz_152 | 2 | 8 | 0 | 2 |
| qz_156 | 3 | 7 | 0 | 3 |
| qz_165 | 8 | 2 | 0 | 3 |
| qz_168 | 1 | 9 | 0 | 3 |
| qz_191 | 8 | 2 | 0 | 2 |
| qz_197 | 9 | 1 | 0 | 3 |
| tc_215 | 1 | 9 | 0 | 2 |
| tc_224 | 1 | 9 | 0 | 3 |
| tc_225 | 5 | 5 | 0 | 2 |
| qz_226 | 9 | 1 | 0 | 2 |
| qz_227 | 2 | 8 | 0 | 2 |
| qz_228 | 5 | 5 | 0 | 2 |
| qz_229 | 9 | 1 | 0 | 2 |
| qz_232 | 1 | 9 | 0 | 3 |
| qz_249 | 1 | 9 | 0 | 6 |
| qz_265 | 2 | 8 | 0 | 2 |
| tc_268 | 8 | 2 | 0 | 2 |
| tc_269 | 1 | 9 | 0 | 5 |
| qz_272 | 8 | 2 | 0 | 2 |
| qz_274 | 3 | 7 | 0 | 4 |
| qz_284 | 1 | 9 | 0 | 2 |
| qz_285 | 8 | 2 | 0 | 2 |
| tc_285 | 4 | 6 | 0 | 2 |
| qz_294 | 4 | 6 | 0 | 3 |
| tc_299 | 3 | 7 | 0 | 2 |
| tc_308 | 6 | 4 | 0 | 3 |
| tc_313 | 5 | 5 | 0 | 2 |
| qz_332 | 6 | 4 | 0 | 3 |
| tc_333 | 2 | 8 | 0 | 3 |
| qz_357 | 4 | 6 | 0 | 2 |
| tc_368 | 1 | 9 | 0 | 2 |
| tc_374 | 1 | 9 | 0 | 3 |
| qz_379 | 3 | 7 | 0 | 2 |
| qz_420 | 1 | 9 | 0 | 6 |
| tc_439 | 1 | 9 | 0 | 3 |
| qz_445 | 1 | 9 | 0 | 7 |
| qz_451 | 9 | 1 | 0 | 3 |
| qz_452 | 6 | 4 | 0 | 2 |
| tc_474 | 4 | 6 | 0 | 2 |
| tc_479 | 1 | 9 | 0 | 3 |
| tc_481 | 6 | 4 | 0 | 2 |
| qz_486 | 5 | 5 | 0 | 4 |
| tc_506 | 7 | 3 | 0 | 2 |
| qz_515 | 1 | 9 | 0 | 2 |
| tc_523 | 2 | 8 | 0 | 2 |
| tc_526 | 8 | 2 | 0 | 3 |
| tc_547 | 6 | 4 | 0 | 3 |
| qz_549 | 4 | 6 | 0 | 2 |
| tc_553 | 4 | 6 | 0 | 2 |
| qz_555 | 5 | 5 | 0 | 2 |
| qz_585 | 3 | 7 | 0 | 3 |
| qz_586 | 1 | 9 | 0 | 6 |
| qz_589 | 7 | 3 | 0 | 2 |
| tc_631 | 1 | 9 | 0 | 2 |
| tc_632 | 1 | 9 | 0 | 3 |
| qz_636 | 3 | 7 | 0 | 2 |
| tc_637 | 1 | 9 | 0 | 2 |
| tc_649 | 6 | 4 | 0 | 2 |
| tc_673 | 9 | 1 | 0 | 2 |
| qz_677 | 8 | 2 | 0 | 2 |
| tc_680 | 6 | 4 | 0 | 2 |
| tc_684 | 4 | 6 | 0 | 3 |
| qz_704 | 3 | 7 | 0 | 5 |
| tc_733 | 1 | 9 | 0 | 7 |
| tc_742 | 7 | 3 | 0 | 3 |
| tc_744 | 5 | 5 | 0 | 2 |
| tc_748 | 9 | 1 | 0 | 2 |
| qz_749 | 2 | 8 | 0 | 5 |
| qz_789 | 1 | 9 | 0 | 2 |
| qz_799 | 9 | 1 | 0 | 2 |
| tc_815 | 8 | 2 | 0 | 2 |
| qz_833 | 3 | 7 | 0 | 2 |
| tc_842 | 6 | 4 | 0 | 2 |
| qz_844 | 2 | 8 | 0 | 3 |
| tc_853 | 3 | 7 | 0 | 4 |
| tc_869 | 6 | 4 | 0 | 3 |
| tc_870 | 8 | 2 | 0 | 2 |
| tc_873 | 3 | 7 | 0 | 2 |
| tc_892 | 1 | 9 | 0 | 2 |
| tc_895 | 4 | 6 | 0 | 2 |
| tc_900 | 1 | 9 | 0 | 2 |
| qz_905 | 2 | 8 | 0 | 3 |
| tc_907 | 1 | 9 | 0 | 2 |
| qz_908 | 2 | 8 | 0 | 2 |
| qz_910 | 3 | 7 | 0 | 4 |
| tc_931 | 6 | 4 | 0 | 3 |
| qz_944 | 5 | 5 | 0 | 3 |
| qz_955 | 9 | 1 | 0 | 2 |
| tc_958 | 6 | 4 | 0 | 2 |
| tc_967 | 8 | 2 | 0 | 2 |
| tc_971 | 1 | 9 | 0 | 3 |
| tc_979 | 1 | 9 | 0 | 4 |
| tc_988 | 2 | 8 | 0 | 3 |
| tc_994 | 1 | 9 | 0 | 5 |
| tc_1014 | 9 | 1 | 0 | 2 |
| tc_1022 | 4 | 6 | 0 | 2 |
| qz_1025 | 1 | 9 | 0 | 5 |
| tc_1031 | 2 | 8 | 0 | 7 |
| tc_1038 | 1 | 9 | 0 | 3 |
| tc_1042 | 8 | 2 | 0 | 3 |
| tc_1045 | 3 | 7 | 0 | 4 |
| tc_1051 | 2 | 8 | 0 | 3 |
| tc_1061 | 2 | 8 | 0 | 4 |
| tc_1071 | 1 | 9 | 0 | 2 |
| tc_1080 | 1 | 9 | 0 | 2 |
| qz_1081 | 1 | 9 | 0 | 5 |
| tc_1088 | 6 | 4 | 0 | 2 |
| qz_1091 | 3 | 7 | 0 | 3 |
| tc_1094 | 1 | 9 | 0 | 5 |
| qz_1096 | 7 | 3 | 0 | 2 |
| qz_1097 | 9 | 1 | 0 | 4 |
| tc_1104 | 2 | 8 | 0 | 4 |
| qz_1109 | 2 | 8 | 0 | 3 |
| qz_1110 | 8 | 2 | 0 | 2 |
| qz_1113 | 9 | 1 | 0 | 3 |
| tc_1118 | 1 | 9 | 0 | 3 |
| tc_1127 | 6 | 4 | 0 | 2 |
| tc_1133 | 6 | 4 | 0 | 3 |
| qz_1138 | 8 | 2 | 0 | 2 |
| qz_1144 | 5 | 5 | 0 | 2 |
| qz_1146 | 4 | 6 | 0 | 2 |
| tc_1149 | 6 | 4 | 0 | 2 |
| qz_1159 | 1 | 9 | 0 | 6 |
| tc_1167 | 3 | 7 | 0 | 4 |
| tc_1168 | 9 | 1 | 0 | 2 |
| tc_1180 | 3 | 7 | 0 | 3 |
| tc_1182 | 2 | 8 | 0 | 2 |
| tc_1184 | 2 | 8 | 0 | 7 |
| qz_1185 | 5 | 5 | 0 | 3 |
| tc_1185 | 2 | 8 | 0 | 3 |
| qz_1196 | 2 | 8 | 0 | 3 |
| tc_1213 | 9 | 1 | 0 | 2 |
| qz_1217 | 3 | 7 | 0 | 2 |
| tc_1226 | 4 | 6 | 0 | 2 |
| tc_1227 | 2 | 8 | 0 | 6 |
| tc_1240 | 6 | 4 | 0 | 2 |
| tc_1252 | 1 | 9 | 0 | 9 |
| qz_1255 | 1 | 9 | 0 | 7 |
| tc_1256 | 6 | 4 | 0 | 3 |
| tc_1265 | 9 | 1 | 0 | 2 |
| qz_1289 | 6 | 4 | 0 | 2 |
| tc_1291 | 4 | 6 | 0 | 2 |
| tc_1296 | 4 | 6 | 0 | 3 |
| qz_1298 | 9 | 1 | 0 | 2 |
| qz_1304 | 8 | 2 | 0 | 3 |
| qz_1310 | 8 | 2 | 0 | 2 |
| qz_1327 | 2 | 8 | 0 | 3 |
| qz_1328 | 3 | 7 | 0 | 2 |
| qz_1331 | 8 | 2 | 0 | 2 |
| tc_1343 | 2 | 8 | 0 | 3 |
| tc_1357 | 7 | 3 | 0 | 2 |
| tc_1369 | 5 | 5 | 0 | 2 |
| qz_1372 | 3 | 7 | 0 | 2 |
| tc_1372 | 2 | 8 | 0 | 2 |
| qz_1374 | 1 | 9 | 0 | 2 |
| tc_1375 | 8 | 2 | 0 | 2 |
| qz_1376 | 8 | 2 | 0 | 2 |
| qz_1389 | 3 | 7 | 0 | 4 |
| qz_1391 | 9 | 1 | 0 | 2 |
| tc_1394 | 3 | 7 | 0 | 2 |
| tc_1406 | 4 | 6 | 0 | 2 |
| qz_1412 | 8 | 2 | 0 | 2 |
| tc_1419 | 1 | 9 | 0 | 5 |
| tc_1444 | 9 | 1 | 0 | 2 |
| qz_1447 | 1 | 9 | 0 | 5 |
| qz_1454 | 3 | 7 | 0 | 4 |
| qz_1459 | 3 | 7 | 0 | 2 |
| tc_1461 | 4 | 6 | 0 | 6 |
| tc_1489 | 3 | 7 | 0 | 2 |
| tc_1491 | 1 | 9 | 0 | 4 |
| tc_1527 | 4 | 6 | 0 | 2 |
| qz_1535 | 8 | 2 | 0 | 4 |
| qz_1549 | 1 | 9 | 0 | 2 |
| tc_1549 | 2 | 8 | 0 | 2 |
| tc_1552 | 4 | 6 | 0 | 3 |
| qz_1553 | 7 | 3 | 0 | 2 |
| qz_1555 | 5 | 5 | 0 | 2 |
| tc_1557 | 1 | 9 | 0 | 2 |
| qz_1562 | 5 | 5 | 0 | 2 |
| tc_1571 | 1 | 9 | 0 | 3 |
| qz_1573 | 6 | 4 | 0 | 2 |
| qz_1601 | 4 | 6 | 0 | 2 |
| qz_1627 | 1 | 9 | 0 | 2 |
| tc_1640 | 3 | 7 | 0 | 6 |
| tc_1651 | 4 | 6 | 0 | 4 |
| tc_1655 | 1 | 9 | 0 | 7 |
| qz_1658 | 5 | 5 | 0 | 8 |
| tc_1668 | 2 | 8 | 0 | 3 |
| qz_1674 | 3 | 7 | 0 | 3 |
| tc_1687 | 1 | 9 | 0 | 3 |
| tc_1689 | 4 | 6 | 0 | 2 |
| qz_1701 | 4 | 6 | 0 | 4 |
| tc_1715 | 8 | 2 | 0 | 2 |
| qz_1731 | 2 | 8 | 0 | 4 |
| tc_1733 | 1 | 9 | 0 | 8 |
| qz_1740 | 6 | 4 | 0 | 2 |
| qz_1742 | 8 | 2 | 0 | 2 |
| tc_1751 | 1 | 9 | 0 | 3 |
| tc_1775 | 8 | 2 | 0 | 5 |
| qz_1776 | 1 | 9 | 0 | 6 |
| qz_1784 | 2 | 8 | 0 | 3 |
| tc_1785 | 3 | 7 | 0 | 3 |
| qz_1788 | 4 | 6 | 0 | 2 |
| qz_1792 | 7 | 3 | 0 | 3 |
| tc_1797 | 2 | 8 | 0 | 2 |
| tc_1799 | 3 | 7 | 0 | 2 |
| qz_1801 | 5 | 5 | 0 | 2 |
| qz_1823 | 5 | 5 | 0 | 2 |
| qz_1840 | 1 | 9 | 0 | 3 |
| tc_1840 | 9 | 1 | 0 | 2 |
| qz_1854 | 8 | 2 | 0 | 3 |
| qz_1876 | 4 | 6 | 0 | 2 |
| qz_1881 | 8 | 2 | 0 | 3 |
| qz_1885 | 3 | 7 | 0 | 3 |
| tc_1885 | 1 | 9 | 0 | 5 |
| tc_1889 | 4 | 6 | 0 | 2 |
| tc_1895 | 1 | 9 | 0 | 6 |
| tc_1904 | 6 | 4 | 0 | 2 |
| qz_1914 | 6 | 4 | 0 | 2 |
| tc_1929 | 3 | 7 | 0 | 2 |
| tc_1948 | 9 | 1 | 0 | 2 |
| tc_1951 | 6 | 4 | 0 | 8 |
| qz_1953 | 2 | 8 | 0 | 2 |
| tc_1955 | 9 | 1 | 0 | 2 |
| tc_1974 | 8 | 2 | 0 | 2 |
| tc_1983 | 1 | 9 | 0 | 4 |
| tc_1985 | 6 | 4 | 0 | 4 |
| tc_1990 | 6 | 4 | 0 | 2 |
| tc_1991 | 3 | 7 | 0 | 3 |
| tc_1992 | 1 | 9 | 0 | 4 |
| tc_1996 | 4 | 6 | 0 | 3 |
| qz_2002 | 3 | 7 | 0 | 2 |
| qz_2008 | 2 | 8 | 0 | 7 |
| tc_2008 | 2 | 8 | 0 | 6 |
| qz_2012 | 8 | 2 | 0 | 2 |
| tc_2016 | 2 | 8 | 0 | 2 |
| tc_2018 | 8 | 2 | 0 | 4 |
| qz_2022 | 9 | 1 | 0 | 2 |
| qz_2025 | 7 | 3 | 0 | 3 |
| qz_2033 | 1 | 9 | 0 | 2 |
| tc_2034 | 8 | 2 | 0 | 3 |
| tc_2039 | 8 | 2 | 0 | 2 |
| tc_2048 | 3 | 7 | 0 | 5 |
| tc_2051 | 1 | 9 | 0 | 5 |
| qz_2055 | 1 | 9 | 0 | 2 |
| qz_2061 | 5 | 5 | 0 | 3 |
| tc_2088 | 4 | 6 | 0 | 4 |
| qz_2094 | 9 | 1 | 0 | 2 |
| qz_2099 | 1 | 9 | 0 | 4 |
| tc_2100 | 2 | 8 | 0 | 2 |
| tc_2104 | 6 | 4 | 0 | 4 |
| tc_2112 | 9 | 1 | 0 | 2 |
| tc_2115 | 9 | 1 | 0 | 2 |
| qz_2116 | 2 | 8 | 0 | 2 |
| qz_2117 | 1 | 9 | 0 | 9 |
| qz_2119 | 1 | 9 | 0 | 2 |
| tc_2157 | 2 | 8 | 0 | 2 |
| tc_2169 | 1 | 9 | 0 | 3 |
| tc_2175 | 9 | 1 | 0 | 2 |
| qz_2191 | 5 | 5 | 0 | 2 |
| qz_2193 | 7 | 3 | 0 | 3 |
| qz_2196 | 3 | 7 | 0 | 2 |
| tc_2197 | 3 | 7 | 0 | 4 |
| tc_2199 | 2 | 8 | 0 | 2 |
| qz_2216 | 4 | 6 | 0 | 2 |
| qz_2219 | 5 | 5 | 0 | 3 |
| qz_2250 | 2 | 8 | 0 | 5 |
| qz_2257 | 1 | 9 | 0 | 2 |
| tc_2267 | 6 | 4 | 0 | 2 |
| tc_2271 | 5 | 5 | 0 | 3 |
| tc_2277 | 2 | 8 | 0 | 4 |
| tc_2284 | 5 | 5 | 0 | 4 |
| qz_2286 | 5 | 5 | 0 | 3 |
| tc_2292 | 9 | 1 | 0 | 2 |
| tc_2302 | 4 | 6 | 0 | 3 |
| tc_2307 | 7 | 3 | 0 | 2 |
| qz_2312 | 9 | 1 | 0 | 2 |
| tc_2312 | 1 | 9 | 0 | 7 |
| qz_2315 | 5 | 5 | 0 | 4 |
| tc_2322 | 2 | 8 | 0 | 2 |
| tc_2337 | 1 | 9 | 0 | 2 |
| qz_2339 | 1 | 9 | 0 | 2 |
| tc_2339 | 3 | 7 | 0 | 3 |
| tc_2355 | 9 | 1 | 0 | 2 |
| tc_2358 | 7 | 3 | 0 | 2 |
| qz_2363 | 8 | 2 | 0 | 2 |
| tc_2368 | 7 | 3 | 0 | 2 |
| tc_2390 | 1 | 9 | 0 | 3 |
| qz_2401 | 9 | 1 | 0 | 2 |
| tc_2405 | 6 | 4 | 0 | 2 |
| qz_2416 | 7 | 3 | 0 | 2 |
| tc_2416 | 9 | 1 | 0 | 2 |
| tc_2419 | 8 | 2 | 0 | 2 |
| tc_2427 | 4 | 6 | 0 | 2 |
| qz_2432 | 9 | 1 | 0 | 2 |
| qz_2434 | 1 | 9 | 0 | 8 |
| tc_2437 | 8 | 2 | 0 | 2 |
| tc_2456 | 9 | 1 | 0 | 2 |
| tc_2463 | 7 | 3 | 0 | 2 |
| tc_2466 | 9 | 1 | 0 | 5 |
| tc_2520 | 6 | 4 | 0 | 2 |
| qz_2528 | 3 | 7 | 0 | 8 |
| tc_2528 | 6 | 4 | 0 | 2 |
| tc_2534 | 1 | 9 | 0 | 2 |
| qz_2545 | 3 | 7 | 0 | 2 |
| qz_2547 | 7 | 3 | 0 | 2 |
| tc_2547 | 4 | 6 | 0 | 6 |
| tc_2567 | 5 | 5 | 0 | 2 |
| tc_2570 | 9 | 1 | 0 | 2 |
| qz_2573 | 2 | 8 | 0 | 4 |
| qz_2574 | 2 | 8 | 0 | 3 |
| tc_2582 | 1 | 9 | 0 | 2 |
| tc_2588 | 2 | 8 | 0 | 2 |
| qz_2592 | 6 | 4 | 0 | 2 |
| tc_2594 | 1 | 9 | 0 | 2 |
| qz_2616 | 2 | 8 | 0 | 7 |
| qz_2617 | 5 | 5 | 0 | 2 |
| qz_2618 | 1 | 9 | 0 | 3 |
| qz_2620 | 5 | 5 | 0 | 6 |
| tc_2620 | 7 | 3 | 0 | 3 |
| qz_2621 | 1 | 9 | 0 | 3 |
| qz_2623 | 2 | 8 | 0 | 2 |
| tc_2625 | 1 | 9 | 0 | 2 |
| tc_2628 | 7 | 3 | 0 | 2 |
| tc_2670 | 2 | 8 | 0 | 3 |
| tc_2682 | 8 | 2 | 0 | 2 |
| tc_2702 | 1 | 9 | 0 | 2 |
| tc_2708 | 3 | 7 | 0 | 2 |
| qz_2715 | 3 | 7 | 0 | 3 |
| qz_2748 | 5 | 5 | 0 | 3 |
| qz_2755 | 4 | 6 | 0 | 3 |
| qz_2756 | 9 | 1 | 0 | 2 |
| tc_2756 | 4 | 6 | 0 | 6 |
| tc_2765 | 5 | 5 | 0 | 2 |
| tc_2774 | 3 | 7 | 0 | 2 |
| tc_2777 | 1 | 9 | 0 | 2 |
| qz_2783 | 3 | 7 | 0 | 2 |
| qz_2792 | 9 | 1 | 0 | 2 |
| tc_2798 | 7 | 3 | 0 | 3 |
| qz_2807 | 8 | 2 | 0 | 2 |
| qz_2811 | 2 | 8 | 0 | 2 |
| tc_2812 | 1 | 9 | 0 | 6 |
| qz_2825 | 7 | 3 | 0 | 3 |
| qz_2829 | 1 | 9 | 0 | 2 |
| tc_2830 | 8 | 2 | 0 | 4 |
| tc_2846 | 9 | 1 | 0 | 2 |
| tc_2858 | 9 | 1 | 0 | 2 |
| qz_2878 | 4 | 6 | 0 | 2 |
| qz_2886 | 1 | 9 | 0 | 8 |
| qz_2890 | 9 | 1 | 0 | 2 |
| qz_2893 | 7 | 3 | 0 | 2 |
| tc_2922 | 6 | 4 | 0 | 2 |
| tc_2925 | 6 | 4 | 0 | 3 |
| tc_2932 | 9 | 1 | 0 | 2 |
| tc_2939 | 3 | 7 | 0 | 4 |
| tc_2985 | 6 | 4 | 0 | 3 |
| tc_3009 | 1 | 9 | 0 | 2 |
| tc_3011 | 5 | 5 | 0 | 7 |
| tc_3035 | 2 | 8 | 0 | 2 |
| tc_3044 | 4 | 6 | 0 | 2 |
| tc_3045 | 1 | 9 | 0 | 2 |
| tc_3065 | 1 | 9 | 0 | 3 |
| tc_3067 | 5 | 5 | 0 | 3 |
| tc_3089 | 7 | 3 | 0 | 3 |
| tc_3105 | 1 | 9 | 0 | 2 |
| tc_3138 | 1 | 9 | 0 | 8 |
| tc_3142 | 6 | 4 | 0 | 3 |

## 6. All-Correct Yield Curve

| Milestone | Cumulative all-correct |
| --- | --- |
| 500 | 154 |
| 1000 | 303 |
| 1500 | 475 |
| 2000 | 808 |
| 2500 | 1043 |
| 3000 | 1234 |
| 3500 | 1435 |

## 7. Judge Agreement Sanity Check on All-Correct

| Metric | Count |
| --- | --- |
| All-correct entries | 1435 |
| All-correct entries with >1 distinct normalized response | 96 |

| QID | Distinct normalized responses | Normalized forms |
| --- | --- | --- |
| tc_1361 | 10 | baby doc duvalier was known for his lavish lifestyle patronage of arts and increasingly authoritarian rule particularly during late 1980s; baby doc duvalier was known for his lavish lifestyle patronage of arts and perceived corruption; “baby doc” duvalier was known for his lavish lifestyle corruption and reliance on duvalier family dynasty; “baby doc” duvalier was known for his lavish lifestyle corruption and reliance on military; “baby doc” duvalier was known for his lavish lifestyle patronage of arts and alleged corruption; “baby doc” duvalier was known for his lavish lifestyle patronage of arts and increasingly authoritarian rule; “baby doc” duvalier was known for his lavish lifestyle patronage of arts and increasingly authoritarian rule particularly in 1980s; “baby doc” duvalier was known for his lavish lifestyle patronage of arts and system of widespread corruption and repression; “baby doc” duvalier was known for his lavish lifestyle patronage of arts and widespread corruption during his presidency; “baby doc” – benevolent dictator known for his lavish lifestyle patronage of arts and use of social programs to maintain popularity |
| tc_2817 | 10 | kangaroos wallabies wombats bilbies quokkas cassowaries bindi echidnas rusa deer fallow deer tasmania red deer tasmania mountain deer new guinea highland; kangaroos wallabies wombats bilbies quokkas echidnas cassowaries tarsiers tree kangaroos possums and various introduced deer species; kangaroos wallabies wombats bilbies quolls echidnas cassowaries bindi tree kangaroos cuscus possums and various rodents; wallabies kangaroos wombats bilbies quokkas echidnas cassowaries and highland hares; wallabies kangaroos wombats bilbies quolls echymiperas possums tarsiers cuscus tree kangaroos muntjacs; wallabies kangaroos wombats bindi echidnas cassowaries rusa deer new guinea; wallabies kangaroos wombats echidnas cassowaries bindi mountain cuscus tree kangaroos bilbies quolls and various possums; wallabies kangaroos wombats possums bilbies quolls echidnas tarsiers; wallabies kangaroos wombats possums quolls bilbies echidnas cassowaries tarsiers; wallabies kangaroos wombats possums tarsiers cuscus tree shrews |
| qz_2566 | 9 | bonnie elizabeth parker and clyde edgar thompson; bonnie elizabeth parker and clyde edgar thompson jr; bonnie elizabeth parker and clyde franklin chang; bonnie elizabeth parker and clyde franklin thompson jr; bonnie elizabeth parker and clyde frederick thompson; bonnie elizabeth parker and clyde harold lee frederick; bonnie elizabeth parker and clyde lee franklin; bonnie parker bonnie elizabeth parker clyde barrow clyde chestnut hunt; bonnie parker bonnie elizabeth parker clyde barrow clyde thomas barrow |
| tc_138 | 6 | henry james hans holbein younger; henry james juan gris; henry james juan ruy ferrer; henry james julio cortázar; henry james julio gonzález; henry james patrick henry green |
| tc_2113 | 6 | alfred stieglitz and edward steichen; alfred stieglitz and geoge young; alfred stieglitz and lothar schuler; alfred stieglitz and lothrup w whitshed; alfred stieglitz and lothrup weschke; alfred stieglitz and lothrup wiggenstein |
| tc_2879 | 6 | during sleep particularly in later stages of night; during sleep particularly in later stages of nrem sleep and during periods of dreaming; during sleep particularly in later stages of nrem sleep and during transition between sleep stages; during sleep particularly in later stages of sleep; during sleep particularly in later stages of sleep cycle; during sleep primarily in later stages of nrem sleep and during transition between sleep stages |
| tc_1937 | 5 | former priest and politician; former priest politician; former priest politician and economist; former priest politician and president; former priest then politician president |
| tc_2251 | 5 | cds; cds and digital downloads; cds and digital music files mp3s etc; compact discs cds; compact discs cds and digital audio players daps mp3 players |
| qz_286 | 4 | three wise men also known as magi; three wise men also known as magi or kings; three wise men or magi; wise men from east |
| tc_1374 | 4 | vertical takeoff and landing vtol aircraft; vertical takeoff and landing vtol jet aircraft; vertical takeoff and landing vtol jet fighter; vtol strike fighter |
| tc_1940 | 4 | barbara fawsett; barbara frank; barbara marx; barbara sinatra |
| tc_2053 | 4 | river shannon shannon hydroelectric dams; river shannon shannon hydroelectric power; shannon river shannon power generation; shannon river shannon power plant |
| tc_2560 | 4 | scutari istanbul; scutari modern day istanbul turkey; scutari present day istanbul turkey; scutari present day üsküdar turkey |
| tc_2831 | 4 | krill small fish and other small crustaceans; krill small fish and other small marine animals; krill small fish and other small marine organisms; krill small fish and other small organisms |
| tc_2917 | 4 | toes and feet; toes and foot; toes and foot arches; toes and part of foot |
| tc_2920 | 4 | insulin cortisol thyroid hormones adrenaline estrogen progesterone testosterone growth hormone melatonin oxytocin erythropoietin antidiuretic hormone parathyroid hormone calcitonin gastrin cholecystokinin; insulin glucagon thyroid hormones t3 t4 cortisol adrenaline epinephrine estrogen progesterone testosterone growth hormone melatonin leptin oxytocin vasopressin; insulin glucagon thyroid hormones t3 t4 cortisol adrenaline epinephrine estrogen progesterone testosterone growth hormone melatonin oxytocin antidiuretic hormone adh parathyroid hormone pth; insulin thyroid hormones t3 t4 cortisol adrenaline epinephrine estrogen progesterone testosterone growth hormone melatonin oxytocin antidiuretic hormone adh parathyroid hormone pth calcitonin |
| tc_2980 | 4 | mohs hardness scale vickers hardness rockwell hardness knoop hardness; mohs scale vickers hardness rockwell hardness knoop hardness; mohs scale vickers hardness rockwell hardness knoop hardness brinell hardness; mohs scale vickers rockwell knoop brinell |
| tc_3132 | 4 | light years parsecs astronomical units; light years parsecs astronomical units au kiloparsecs kpc megaparsecs mpc; light years parsecs astronomical units au kiloparsecs megaparsecs; parsecs light years astronomical units |
| qz_69 | 3 | 212; 212 °f; 212°f |
| tc_1126 | 3 | judgement day; judgment day; terminator 2 judgment day |

## 8. Downstream Usability

| Metric | Count |
| --- | --- |
| Total passing entries | 3115 |
| Passing-true | 1435 |
| Passing-false | 1680 |

## Totals

| Metric | Count |
| --- | --- |
| Total responses | 35000 |
| Expected responses | 35000 |

