
class create_data_goal():
    def __init__(self, dialogue):
       #print("len(dialogue['turns'])::",len(dialogue['turns']))
       #print("dialogue['turns'][len(dialogue['turns'])-2]['state']::",dialogue['turns'][len(dialogue['turns'])-2]['state'])
       self.state = dialogue['turns'][len(dialogue['turns'])-2]['state']
       pass

    def transform_goal(self, source_goal):
        """
        Transforms the source goal format into the destination goal format, handling "book" and "book_again" slots.
        For booking-related slots, splits values into "book" and "book_again". For other two-value slots, retains the second value.

        Args:
            source_goal (dict): The source goal dictionary.

        Returns:
            dict: The transformed goal dictionary in the destination format.
        """
        destination_goal = {}

        # Handle 'inform' section
        if "inform" in source_goal:
            for domain, slots in source_goal["inform"].items():
                if domain not in destination_goal:
                    destination_goal[domain] = {}

                # Split into info, book, and book_again sections
                info = {}
                book = {}
                book_again = {}
                for slot, value in slots.items():
                    formatted_slot = slot.replace("price range", "pricerange").replace("arrive by", "arriveBy").replace("leave at", "leaveAt").replace('train id','trainID')
                    if domain == 'taxi' and slot == 'type':
                        formatted_slot = 'car type'						
                    # Ignore slots with "yes|no" values
                    if value == "no|yes" or value == "yes|no":
                        continue

                    # Handle slots with two values separated by '|'
                    if "|" in value:
                        values = value.split("|")
                        first_value, second_value = values[0], values[1]
                        # For book-related slots, split into "book" and "book_again"
                        if slot.startswith("book "):
                            book_key = slot.replace("book ", "")
                            book[book_key] = first_value
                            book_again[book_key] = second_value
                        else:
                            # For non-book slots
                            if first_value == second_value:
                                info[formatted_slot] = first_value
                            else:
                                if self.state[domain][slot]:
                                    info[formatted_slot] = self.state[domain][slot]
                                else:
                                    info[formatted_slot] = second_value
                    else:
                        # Handle single values
                        if slot.startswith("book "):
                            book_key = slot.replace("book ", "")
                            book[book_key] = value
                            if book_again:
                                book_again[book_key] = value
                        else:
                            #formatted_slot = slot.replace("price range", "pricerange").replace("arrive by", "arriveBy")
                            info[formatted_slot] = value

                if info:
                    destination_goal[domain]["info"] = info
                if book:
                    destination_goal[domain]["book"] = book
                if book_again:
                    destination_goal[domain]["book_again"] = book_again
                if book:
                    destination_goal[domain]["booked"] = "?"

        # Handle 'request' section
        if "request" in source_goal:
            for domain, slots in source_goal["request"].items():
                if domain not in destination_goal:
                    destination_goal[domain] = {}

                # Add requested slots
                reqt = {}
                for slot in slots:
                    formatted_slot = slot.replace("price range", "pricerange").replace("arrive by", "arriveBy").replace("leave at", "leaveAt").replace('train id','trainID')
                    if domain == 'taxi' and slot == 'type':
                        formatted_slot = 'car type'
                    reqt[formatted_slot] = "?"

                if reqt:
                    destination_goal[domain]["reqt"] = reqt

        return destination_goal

