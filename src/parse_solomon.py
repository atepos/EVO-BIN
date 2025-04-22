"""
parse_solomon.py

Parser instancí VRPTW ve Solomonově XML formátu:
  - parse_solomon_xml – načte XML soubor (např. R201_025.xml) a vrátí dict s klíči:
      - vehicle_capacity: float
      - vehicle_speed: float
      - depot: dict (id, x, y, tw_start, tw_end, service_time)
      - customers: list of dict (id, x, y, demand, tw_start, tw_end, service_time)

Závislosti:
  xml.etree.ElementTree

Autor:      Petr Kaška
Vytvořeno:  2025-04-22
"""
import xml.etree.ElementTree as ET

def parse_solomon_xml(filename):
    """
    Načte VRPTW instanci ze souboru ve Solomonovském formátu XML,
    např. R201_025.xml, a vrátí slovník:
      {
        'vehicle_capacity': float,
        'vehicle_speed': float,
        'depot': {...},
        'customers': [...],
      }
    """

    tree = ET.parse(filename)
    root = tree.getroot()

    fleet_element = root.find("fleet/vehicle_profile")
    vehicle_capacity = float(fleet_element.find("capacity").text)
    departure_node_id = int(fleet_element.find("departure_node").text)
    arrival_node_id   = int(fleet_element.find("arrival_node").text)
    
    node_dict = {}
    node_elements = root.findall("network/nodes/node")
    for ne in node_elements:
        nid = int(ne.attrib["id"])
        cx = float(ne.find("cx").text)
        cy = float(ne.find("cy").text)
        node_dict[nid] = (cx, cy)

    if departure_node_id not in node_dict:
        raise ValueError(f"Depot (node_id={departure_node_id}) nebyl nalezen mezi nody.")

    depot_x, depot_y = node_dict[departure_node_id]
    depot = {
        'id': departure_node_id,
        'x': depot_x,
        'y': depot_y,
        'tw_start': 0.0,
        'tw_end': 999999.0,   
        'service_time': 0.0
    }

    customers = []
    requests_elem = root.findall("requests/request")
    for req in requests_elem:
        req_id = int(req.attrib["id"])       
        node_id = int(req.attrib["node"])    
        quantity = float(req.find("quantity").text)
        service_time = float(req.find("service_time").text)

        tw_start = float(req.find("tw/start").text)
        tw_end   = float(req.find("tw/end").text)

        if node_id not in node_dict:
            continue  

        x_, y_ = node_dict[node_id]
        cust_dict = {
            'id': req_id,
            'x': x_,
            'y': y_,
            'demand': quantity,
            'tw_start': tw_start,
            'tw_end': tw_end,
            'service_time': service_time
        }
        customers.append(cust_dict)

    vrptw_instance = {
        'vehicle_capacity': vehicle_capacity,
        'vehicle_speed': 1.0,  
        'depot': depot,
        'customers': customers
    }
    return vrptw_instance
