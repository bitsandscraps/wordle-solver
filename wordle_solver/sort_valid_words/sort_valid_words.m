validWords = Map[ToString, ReadList["word_list.txt"]]
frequencies = (WordFrequencyData[#, IgnoreCase->True])&/@validWords
frequencies = ReplacePart[frequencies, Map[First, Position[frequencies, Missing]]->0]
ordering = Reverse[Ordering[frequencies]]
Export["sorted_word_list.txt", Extract[validWords, Map[List, ordering]], "List"]
