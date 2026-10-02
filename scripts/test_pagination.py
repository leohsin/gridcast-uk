from gridcast_uk.data.neso import fetch_demand_records


page_1 = fetch_demand_records(
    limit=5,
    offset=0,
)

page_2 = fetch_demand_records(
    limit=5,
    offset=5,
)


records_1 = page_1["result"]["records"]
records_2 = page_2["result"]["records"]


print("Page 1 IDs:")
for record in records_1:
    print(record["_id"])


print("\nPage 2 IDs:")
for record in records_2:
    print(record["_id"])