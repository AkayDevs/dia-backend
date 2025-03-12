import cv2
import os
import argparse

def draw_bbox(image_path, coordinates, output_path=None, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on an image and save the result.
    
    Args:
        image_path (str): Path to the input image
        coordinates (list): List of bounding box coordinates in format [[x1, y1, x2, y2], ...]
        output_path (str, optional): Path to save the output image. If None, saves in the same directory
                                    with '_bbox' suffix
        color (tuple, optional): BGR color for the bounding box. Default is green (0, 255, 0)
        thickness (int, optional): Thickness of the bounding box lines. Default is 2
    
    Returns:
        str: Path to the saved image
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Make a copy to avoid modifying the original
    image_with_bbox = image.copy()
    
    # Draw each bounding box
    for bbox in coordinates:
        if len(bbox) != 4:
            raise ValueError(f"Invalid bounding box format: {bbox}. Expected [x1, y1, x2, y2]")
        
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image_with_bbox, (x1, y1), (x2, y2), color, thickness)
    
    # Generate output path if not provided
    if output_path is None:
        filename, ext = os.path.splitext(image_path)
        output_path = f"{filename}_bbox{ext}"
    
    # Save the image with bounding boxes
    cv2.imwrite(output_path, image_with_bbox)
    print(f"Image with bounding boxes saved to {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Draw bounding boxes on an image")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--coordinates", nargs="+", action="append", type=int,
                        help="Bounding box coordinates in format 'x1 y1 x2 y2'. Can be specified multiple times.")
    parser.add_argument("--output", help="Path to save the output image. Default is input_bbox.ext")
    parser.add_argument("--color", nargs=3, type=int, default=[0, 255, 0],
                        help="BGR color for bounding boxes. Default is green (0, 255, 0)")
    parser.add_argument("--thickness", type=int, default=2,
                        help="Thickness of bounding box lines. Default is 2")
    
    args = parser.parse_args()
    
    # Process coordinates
    coordinates = []
    if args.coordinates:
        for coord_set in args.coordinates:
            if len(coord_set) != 4:
                raise ValueError(f"Invalid coordinates: {coord_set}. Expected 4 values (x1, y1, x2, y2)")
            coordinates.append(coord_set)
    
    # Draw bounding boxes and save the image
    draw_bbox(args.image_path, coordinates, args.output, tuple(args.color), args.thickness)

if __name__ == "__main__":
    main()
