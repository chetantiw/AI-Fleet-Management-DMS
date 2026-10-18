"""Ground-truth labels from frame-by-frame review (1 frame per second) of the Oct 13 bus videos.
Each entry: (start_s, end_s). 'side' = head clearly turned to window / passenger / back.
'closed' = eyes shut (acted drowsiness). 'drink' = bottle at the mouth or lifted to drink.
'call' = phone held at the ear. 'hand' = phone held in hand / lap. 'absent' = no driver in seat.
'sunglasses' = wearing dark glasses. 'down' = looking down normally (not drowsy, no phone)."""
L = {
 "IMG_0110": dict(
   side=[(1,2),(6,7),(13,13),(16,16),(82,82),(92,92),(100,100),(105,105),(108,108),(120,120),(134,136),(145,145),(151,155),(164,169),(187,189),(196,198),(202,202),(205,205),(210,210),(215,215),(225,225),(228,228)],
   closed=[(63,63),(66,68),(72,79)],
   drink=[(97,98),(103,103),(123,123)],
   call=[(38,57)], hand=[], absent=[(250,254)], sunglasses=[], down=[(22,37),(84,91),(109,119),(138,144),(230,245)]),
 "IMG_0107": dict(
   side=[(10,12),(14,17),(23,27),(37,40),(58,58),(64,64),(66,66),(72,73),(85,85),(110,113),(120,131),(137,137),(139,140),(147,147),(150,150),(163,165),(176,176),(187,191),(235,239)],
   closed=[],
   drink=[(67,71),(74,76),(79,82),(84,84)],
   call=[], hand=[], absent=[(0,4)], sunglasses=[(153,235)], down=[]),
 "IMG_0112": dict(
   side=[(13,14),(19,23),(27,27),(30,31),(39,39),(41,43),(46,47),(50,51),(67,67),(72,73),(76,76),(78,78),(82,82),(98,101),(110,113),(137,138),(155,158),(164,164),(197,202),(208,208),(250,250),(260,262),(269,272),(280,280),(289,289),(291,291),(294,294),(300,303)],
   closed=[(124,133),(136,136)],
   drink=[(94,98),(102,104)],
   call=[(166,193)], hand=[(150,165),(194,208)], absent=[(0,5)], sunglasses=[(247,314)], down=[(210,245)]),
}
