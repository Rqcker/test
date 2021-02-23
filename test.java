for(int i = 0; i < word.length(); i++) {
			arr = word.toCharArray();
			for (char alphabet = 'a';  alphabet <= 'z'; alphabet++) {
				arr[i] = alphabet;
				str = String.valueOf(arr);
				if(dict.wordExists(str)) {
					try {
						suggestion.insWord(str);
					} catch (SpellCheckException e) {
						// System.err.println(e.getMessage());
					}
				}
			}
		}


@Override
	public boolean wordExists(String word) {
		int test = giveHashCode(word) % size;
		int dHash = 5 - (giveHashCode(word) % 5);
		numberOfOperations++;
		numberOfProbes++;
		
		//Checks the cell pointed by hashCode, if hCode cell of W is empty, returns false.
		while (hTable[test] != null) {
			
			if (hTable[test].equals(word)) { //if given word is equal, returns true.
				return true;
			}
			test = (test + dHash) % size;
			numberOfProbes++; //Increase the number of probes each time a cell is visited
		}
		return false;
	}
