
(*note that in the program below, the list of reduction steps given by
mreduce has many duplications.  This is very easy to avoid by
monitoring l2,l3,l4, l5,l6, etc, in mreduce and the print functions.
This is your job to do. *)

datatype LEXP =  APP of LEXP * LEXP | LAM of string *  LEXP |  ID of string;

(*Prints a term*)
fun printLEXP (ID v) =
    print v
  | printLEXP (LAM (v,e)) =
    (print "(\\";
     print v;
     print ".";
     printLEXP e;
     print ")")
  | printLEXP (APP(e1,e2)) =
    (print "(";
     printLEXP e1;
     print " ";
     printLEXP e2;
     print ")");

fun is_var (ID id) =  true |
    is_var _ = false;


(*the function below adds lambda id to a list of terms *)
fun addlam id [] = [] |
    addlam id (e::l) = (LAM(id,e))::(addlam id l);

(*similar to above but adds app backward *)
fun addbackapp [] e2 = []|
    addbackapp (e1::l) e2 = (APP(e1,e2)):: (addbackapp l e2);

(*similar to above but adds app forward *)
fun addfrontapp e1 [] = []|
    addfrontapp e1  (e2::l) = (APP(e1,e2)):: (addfrontapp e1 l);

(*prints elements from a list putting an arrow in between*)
fun printlistreduce [] = ()|
    printlistreduce (e::[]) = (printLEXP e) |
    printlistreduce (e::l) = (printLEXP e; print "-->" ; (printlistreduce l));


val vx = (ID "x");
val vy = (ID "y");
val vz = (ID "z");
val t1 = (LAM("x",vx));
val t2 = (LAM("y",vx));
val t3 = (APP(APP(t1,t2),vz));
val t4 = (APP(t1,vz));
val t5 = (APP(t3,t3));
val t6 = (LAM("x",(LAM("y",(LAM("z",(APP(APP(vx,vz),(APP(vy,vz))))))))));
val t7 = (APP(APP(t6,t1),t1));
val t8 = (LAM("z", (APP(vz,(APP(t1,vz))))));
val t9 = (APP(t8,t3));



(* checks whether a variable is free in a term *)

fun free id1 (ID id2) = if (id1 = id2) then  true else false|
    free id1 (APP(e1,e2))= (free id1 e1) orelse (free id1 e2) | 
    free id1 (LAM(id2, e1)) = if (id1 = id2) then false else (free id1 e1);

(* finds new variables which are fresh  in l and different from id*)
    
fun findme id l = (let val id1 = id^"1"  in if not (List.exists (fn x => id1 = x) l) then id1 else (findme id1 l) end);


(* finds the list of free variables in a term *)

fun freeVars (ID id2)       = [id2]
  | freeVars (APP(e1,e2))   = freeVars e1 @ freeVars e2
  | freeVars (LAM(id2, e1)) = List.filter (fn x => not (x = id2)) (freeVars e1);


(*does substitution avoiding the capture of free variables*)

fun subs e id (ID id1) =  if id = id1 then e else (ID id1) |
    subs e id (APP(e1,e2)) = APP(subs e id e1, subs e id e2)|
    subs e id (LAM(id1,e1)) = (if id = id1 then LAM(id1,e1) else
                                   if (not (free id e1) orelse not (free id1 e))
				       then LAM(id1,subs e id e1)
                                   else (let val id2 = (findme id ([id1]@ (freeVars e) @ (freeVars e1)))
					 in LAM(id2, subs e id (subs (ID id2) id1 e1)) end));

datatype BEXP =
    BAPP of BEXP * BEXP | BLAM of BEXP |  BID of int;

(*Prints a term in classical lambda calculus with de Bruijn indices*)
fun printBEXP (BID n) =
    print (Int.toString n)
  | printBEXP (BLAM (e)) =
    (print "(\\";
     printBEXP e;
     print ")")
  | printBEXP (BAPP(e1,e2)) =
    (print "(";
     printBEXP e1;
     print " ";
     printBEXP e2;
     print ")");
    

datatype IEXP =  IAPP of IEXP * IEXP | ILAM of string *  IEXP |  IID of string;

(*Prints a Iterm*)
fun printIEXP (IID v) =
    print v
  | printIEXP (ILAM (v,e)) =
    (print "[";
     print v;
     print "]";
     printIEXP e)
  | printIEXP (IAPP(e1,e2)) =
    (print "<";
     printIEXP e1;
     print ">";
     printIEXP e2);

