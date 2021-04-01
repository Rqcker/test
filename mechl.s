.data

	.equ LEN, 3
	.equ COLORS, 3
	.equ NAN1, 8
	.equ NAN2, 9

	.equ BUTTONBASE,	0xFF200050
	.equ HEXBASE,	0xFF200020
	.equ BUTTON_NO, 1	

.align 1	
digits:
	.byte  0b0111111	@ 0
	.byte  0b0000110	@ 1
	.byte  0b1011011	@ 2
	.byte  0b1001111	@ 3
	.byte  0b1100110	@ 4
	.byte  0b1101101	@ 5
	.byte  0b1111100	@ 6
	.byte  0b0000111	@ 7
	.byte  0b1111111	@ 8
	.byte  0b1101111	@ 9

.align 1
char_o:	.byte  0b1011100	@ o
char_n:	.byte  0b1010100	@ n
char_f:	.byte  0b1110001	@ f
	
.align 4
	@ a simple test string
teststr:	@ 9 words, 43 chars
	.asciz "The quick brown fox jumps over the lazy dog"

	
.align 4
secret:.word 1 
	.word 2 
	.word 1 

.align 4
guess:	.word 3 
	.word 1 
	.word 3 
@ Expect Answer: 0 1
.align 4
expect: .byte 0
	.byte 1
