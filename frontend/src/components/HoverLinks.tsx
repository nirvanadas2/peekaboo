const HoverLinks = ({ text }: { text: string }) => {
  return (
    <div className="hover-link">
      <div className="hover-in">
        {text} <div>{text}</div>
      </div>
    </div>
  );
};

export default HoverLinks;
