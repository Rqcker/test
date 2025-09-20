# -*- coding: utf-8 -*- #
# Author: yinzhangyue
# Created: 2024/6/10
GSM8K_CoT_Prompt = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. So the answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. So the answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. So the answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. So the answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. So the answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. So the answer is 8.

Q: {}
A: """

AQuA_CoT_Prompt = """Q: John found that the average of 15 numbers is 40. If 10 is added to each number then the mean of the numbers is?
Answer Choices: (A) 50 (B) 45 (C) 65 (D) 78 (E) 64
A: If 10 is added to each number, then the mean of the numbers also increases by 10. So the new mean would be 50. So the answer is (A).

Q: If a / b = 3/4 and 8a + 5b = 22, then find the value of a.
Answer Choices: (A) 1/2 (B) 3/2 (C) 5/2 (D) 4/2 (E) 7/2
A: If a / b = 3/4, then b = 4a / 3. So 8a + 5(4a / 3) = 22. This simplifies to 8a + 20a / 3 = 22, which means 44a / 3 = 22. So a is equal to 3/2. So the answer is (B).

Q: A person is traveling at 20 km/hr and reached his destiny in 2.5 hr then find the distance?
Answer Choices: (A) 53 km (B) 55 km (C) 52 km (D) 60 km (E) 50 km
A: The distance that the person traveled would have been 20 km/hr * 2.5 hrs = 50 km. So the answer is (E).

Q: How many keystrokes are needed to type the numbers from 1 to 500?
Answer Choices: (A) 1156 (B) 1392 (C) 1480 (D) 1562 (E) 1788
A: There are 9 one-digit numbers from 1 to 9. There are 90 two-digit numbers from 10 to 99. There are 401 three-digit numbers from 100 to 500. 9 + 90(2) + 401(3) = 1392. So the answer is (B).

Q: {}
A: """

CSQA_CoT_Prompt = """Q: What do people use to absorb extra ink from a fountain pen? 
Answer Choices: (A) shirt pocket (B) calligrapher's hand (C) inkwell (D) desk drawer (E) blotter
A: The answer must be an item that can absorb ink. Of the above choices, only blotters are used to absorb ink. So the answer is (E).

Q: What home entertainment equipment requires cable?
Answer Choices: (A) radio shack (B) substation (C) television (D) cabinet
A: The answer must require cable. Of the above choices, only television requires cable. So the answer is (C).

Q: The fox walked from the city into the forest, what was it looking for? 
Answer Choices: (A) pretty flowers (B) hen house (C) natural habitat (D) storybook
A: The answer must be something in the forest. Of the above choices, only natural habitat is in the forest. So the answer is (B).

Q: Sammy wanted to go to where the people were. Where might he go? 
Answer Choices: (A) populated areas (B) race track (C) desert (D) apartment (E) roadblock
A: The answer must be a place with a lot of people. Of the above choices, only populated areas have a lot of people. So the answer is (A).

Q: Where do you put your grapes just before checking out? 
Answer Choices: (A) mouth (B) grocery cart (C) supermarket (D) fruit basket (E) fruit market
A: The answer should be the place where grocery items are placed before checking out. Of the above choices, grocery cart makes the most sense for holding grocery items. So the answer is (B).

Q: Google Maps and other highway and street GPS services have replaced what? 
Answer Choices: (A) united states (B) mexico (C) countryside (D) atlas
A: The answer must be something that used to do what Google Maps and GPS services do, which is to give directions. Of the above choices, only atlases are used to give directions. So the answer is (D).

Q: Before getting a divorce, what did the wife feel who was doing all the work? 
Answer Choices: (A) harder (B) anguish (C) bitterness (D) tears (E) sadness
A: The answer should be the feeling of someone getting divorced who was doing all the work. Of the above choices, the closest feeling is bitterness. So the answer is (C).

Q: {}
A: """

Strategy_CoT_Prompt = """Q: Do hamsters provide food for any animals?
A: Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals. So the answer is yes.

Q: Could Brooke Shields succeed at University of Pennsylvania?
A: Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania. So the answer is yes.

Q: Yes or no: Hydrogen’s atomic number squared exceeds number of Spice Girls?
A: Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen's atomic number squared is less than 5. So the answer is no.

Q: Yes or no: Is it common to see frost during some college commencements?
A: College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements. So the answer is yes.

Q: Yes or no: Could a llama birth twice during War in Vietnam (1945-46)?
A: The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam. So the answer is no.

Q: Yes or no: Would a pear sink in water?
A: The density of a pear is about 0:6g=cm3, which is less than water. Objects less dense than water float. Thus, a pear would float. So the answer is no.

Q: {}
A: """

GSM8K_Prompt = "Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nA: Let's think step by step.\n1. Initially, there are 15 trees in the grove.\n2. After planting, there will be 21 trees in total.\n3. To find how many trees were planted, subtract the initial number from the final number:21 - 15 = 6 trees.\nThus, the answer is 6.\nQ: {}\nA: Let's think step by step."
AQuA_Prompt = "Q: John found that the average of 15 numbers is 40. If 10 is added to each number then the mean of the numbers is?\nAnswer Choices: (A) 50 (B) 45 (C) 65 (D) 78 (E) 64\nA: Let's think step by step.\n1. Initial average = 40\n2. When 10 is added to each number:\n* Each number increases by 10\n* The average will also increase by 10\n3. New average = 40 + 10 = 50\nThus, the answer is (A).\nQ: {}\nA: Let's think step by step."
CSQA_Prompt = "Q: What do people use to absorb extra ink from a fountain pen?\nAnswer Choices: (A) shirt pocket (B) calligrapher's hand (C) inkwell (D) desk drawer (E) blotter\nA: Let's think step by step.\n1. A blotter is specifically designed to absorb excess ink from writing\n2. Other options are either ink sources (inkwell) or would damage items (shirt, desk)\n3. 'Calligrapher's hand' is not an ink-absorbing tool\nThus, the answer is (E).\nQ: {}\nA: Let's think step by step."
StrategyQA_Prompt = "Q: Do hamsters provide food for any animals?\nA: Let's think step by step.\n1. Hamsters are small rodents, making them prey animals\n2. Natural predators of hamsters include: Owls, Snakes, Foxes, Wild cats\n3. In the wild, hamsters serve as a food source for these predators\n4. Even domestic hamsters can be prey to cats, dogs, or birds if not protected\nThus, the answer is yes.\nQ: {}\nA: Let's think step by step."

GSM8K_Prompt_list = [{"question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?", "reasoning": "Let's think step by step.\n1. Initially, there are 15 trees in the grove.\n2. After planting, there will be 21 trees in total.\n3. To find how many trees were planted, subtract the initial number from the final number:21 - 15 = 6 trees.", "answer": "6."}, {"question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?", "reasoning": "Let's think step by step.\n1. Initially, there are 3 cars in the parking lot.\n2. 2 more cars arrive.\n3. Total cars now = 3 + 2 = 5.", "answer": "5."}, {"question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?", "reasoning": "Let's think step by step.\n1. Leah had 32 chocolates.\n2. Her sister had 42 chocolates.\n3. Total chocolates = 32 + 42 = 74.\n4. They ate 35 chocolates.\n5. Chocolates left = 74 - 35 = 39.", "answer": "39."}, {"question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?", "reasoning": "Let's think step by step.\n1. Jason started with 20 lollipops.\n2. After giving some to Denny, he has 12 left.\n3. Lollipops given = 20 - 12 = 8.", "answer": "8."}, {"question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?", "reasoning": "Let's think step by step.\n1. Shawn started with 5 toys.\n2. He received 2 toys from his mom and 2 from his dad.\n3. Total new toys = 2 + 2 = 4.\n4. Total toys now = 5 + 4 = 9.", "answer": "9."}, {"question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?", "reasoning": "Let's think step by step.\n1. Initially, there were 9 computers.\n2. From Monday to Thursday is 4 days.\n3. Computers added = 5 per day × 4 days = 20.\n4. Total computers now = 9 + 20 = 29.", "answer": "29."}, {"question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?", "reasoning": "Let's think step by step.\n1. Michael started with 58 golf balls.\n2. Lost 23 on Tuesday: 58 - 23 = 35.\n3. Lost 2 on Wednesday: 35 - 2 = 33.", "answer": "33."}, {"question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?", "reasoning": "Let's think step by step.\n1. Olivia has $23.\n2. Cost per bagel = $3.\n3. Total cost = 5 × $3 = $15.\n4. Money left = $23 - $15 = $8.", "answer": "8."}]
AQuA_Prompt_list = [{"question": "John found that the average of 15 numbers is 40. If 10 is added to each number then the mean of the numbers is?\nAnswer Choices: (A) 50 (B) 45 (C) 65 (D) 78 (E) 64", "reasoning": "Let's think step by step.\n1. Initial average = 40\n2. When 10 is added to each number:\n* Each number increases by 10\n* The average will also increase by 10\n3. New average = 40 + 10 = 50", "answer": "(A)."}, {"question": "If a / b = 3/4 and 8a + 5b = 22, then find the value of a.\nAnswer Choices: (A) 1/2 (B) 3/2 (C) 5/2 (D) 4/2 (E) 7/2", "reasoning": "Let's think step by step.\n1. Given a/b = 3/4, so b = (4/3)a.\n2. Substitute b into 8a + 5b = 22:\n   - 8a + 5*(4a/3) = 22\n3. Simplify:\n   - 8a + (20a/3) = 22\n   - Multiply both sides by 3 to eliminate fraction:\n     24a + 20a = 66\n     44a = 66\n4. Solve for a:\n   - a = 66/44 = 3/2", "answer": "(B)."}, {"question": "A person is traveling at 20 km/hr and reached his destiny in 2.5 hr then find the distance?\nAnswer Choices: (A) 53 km (B) 55 km (C) 52 km (D) 60 km (E) 50 km", "reasoning": "Let's think step by step.\n1. Speed = 20 km/hr\n2. Time = 2.5 hr\n3. Distance = Speed × Time = 20 km/hr × 2.5 hr = 50 km", "answer": "(E)."}, {"question": "How many keystrokes are needed to type the numbers from 1 to 500?\nAnswer Choices: (A) 1156 (B) 1392 (C) 1480 (D) 1562 (E) 1788", "reasoning": "Let's think step by step.\n1. Numbers 1-9 (one-digit numbers): 9 numbers × 1 digit = 9 keystrokes\n2. Numbers 10-99 (two-digit numbers): 90 numbers × 2 digits = 180 keystrokes\n3. Numbers 100-500 (three-digit numbers): 401 numbers × 3 digits = 1203 keystrokes\n4. Total keystrokes = 9 + 180 + 1203 = 1392", "answer": "(B)."}]
CSQA_Prompt_list = [{"question": "What do people use to absorb extra ink from a fountain pen?\nAnswer Choices: (A) shirt pocket (B) calligrapher's hand (C) inkwell (D) desk drawer (E) blotter", "reasoning": "Let's think step by step.\n1. A blotter is specifically designed to absorb excess ink from writing\n2. Other options are either ink sources (inkwell) or would damage items (shirt, desk)\n3. 'Calligrapher's hand' is not an ink-absorbing tool", "answer": "(E)."}, {"question": "What home entertainment equipment requires cable?\nAnswer Choices: (A) radio shack (B) substation (C) television (D) cabinet", "reasoning": "Let's think step by step.\n1. We need equipment that requires cable.\n2. (A) Radio Shack is a store.\n3. (B) Substation is part of electrical infrastructure.\n4. (C) Television often requires cable service.\n5. (D) Cabinet is furniture.", "answer": "(C)."}, {"question": "The fox walked from the city into the forest, what was it looking for?\nAnswer Choices: (A) pretty flowers (B) hen house (C) natural habitat (D) storybook", "reasoning": "Let's think step by step.\n1. A fox's natural environment is the forest.\n2. It is likely seeking its natural habitat.\n3. Among the options, (C) natural habitat fits best.", "answer": "(C)."}, {"question": "Sammy wanted to go to where the people were. Where might he go?\nAnswer Choices: (A) populated areas (B) race track (C) desert (D) apartment (E) roadblock", "reasoning": "Let's think step by step.\n1. Sammy wants to be where there are many people.\n2. (A) Populated areas have many people.\n3. (B) Race tracks have people during events.\n4. (C) Desert is sparsely populated.\n5. (D) Apartment has limited people.\n6. (E) Roadblock is not a gathering place.", "answer": "(A)."}, {"question": "Where do you put your grapes just before checking out?\nAnswer Choices: (A) mouth (B) grocery cart (C) supermarket (D) fruit basket (E) fruit market", "reasoning": "Let's think step by step.\n1. Before checking out, items are in a container.\n2. (A) Mouth is for eating, not appropriate here.\n3. (B) Grocery cart holds items while shopping.\n4. (C) Supermarket is the store itself.\n5. (D) Fruit basket is typically at home.\n6. (E) Fruit market is another store.", "answer": "(B)."}, {"question": "Google Maps and other highway and street GPS services have replaced what?\nAnswer Choices: (A) united states (B) mexico (C) countryside (D) atlas", "reasoning": "Let's think step by step.\n1. GPS services provide maps and directions.\n2. (D) Atlas is a book of maps used before GPS.\n3. (A), (B), and (C) are places, not tools.", "answer": "(D)."}, {"question": "Before getting a divorce, what did the wife feel who was doing all the work?\nAnswer Choices: (A) harder (B) anguish (C) bitterness (D) tears (E) sadness", "reasoning": "Let's think step by step.\n1. Doing all the work may lead to resentment.\n2. (C) Bitterness reflects resentment from unfairness.\n3. (B) Anguish and (E) sadness are general distress.", "answer": "(C)."}]
StrategyQA_Prompt_list = [{"question": "Do hamsters provide food for any animals?", "reasoning": "Let's think step by step.\n1. Hamsters are small rodents, making them prey animals\n2. Natural predators of hamsters include: Owls, Snakes, Foxes, Wild cats\n3. In the wild, hamsters serve as a food source for these predators\n4. Even domestic hamsters can be prey to cats, dogs, or birds if not protected", "answer": "yes."}, {"question": "Could Brooke Shields succeed at University of Pennsylvania?", "reasoning": "Let's think step by step.\n1. Brooke Shields graduated from Princeton University.\n2. Princeton and UPenn are both Ivy League schools.\n3. They have similar academic rigor.\n4. Therefore, she could succeed at UPenn.", "answer": "yes."}, {"question": "Yes or no: Hydrogen’s atomic number squared exceeds number of Spice Girls?", "reasoning": "Let's think step by step.\n1. Hydrogen's atomic number is 1.\n2. Squared, that's 1^2 = 1.\n3. There are 5 Spice Girls.\n4. 1 is less than 5.", "answer": "no."}, {"question": "Yes or no: Is it common to see frost during some college commencements?", "reasoning": "Let's think step by step.\n1. Some commencements are held in December.\n2. December is winter in many regions.\n3. Frost is common in winter.\n4. Therefore, frost can occur during some commencements.", "answer": "yes."}, {"question": "Yes or no: Could a llama birth twice during War in Vietnam (1945-46)?", "reasoning": "Let's think step by step.\n1. The war lasted about 6 months.\n2. Llama gestation period is ~11 months.\n3. It's impossible for two births in 6 months.", "answer": "no."}, {"question": "Yes or no: Would a pear sink in water?", "reasoning": "Let's think step by step.\n1. Water density is 1.0 g/cm³.\n2. Pear density is ~0.6 g/cm³.\n3. Objects less dense than water float.\n4. Therefore, a pear would float.", "answer": "no."}]


if __name__ == "__main__":
    print("GSM8K_Prompt_list: ", GSM8K_Prompt_list)
    print("AQuA_Prompt_list: ", AQuA_Prompt_list)
    print("CSQA_Prompt_list: ", CSQA_Prompt_list)
    print("StrategyQA_Prompt: ", StrategyQA_Prompt)
