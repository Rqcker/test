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
