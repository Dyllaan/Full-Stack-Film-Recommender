import PropTypes from  'prop-types';
/**
 * @author Louis Figes
 * @generated GitHub Copilot was used in the creation of this code.
 * https://www.w3schools.com/howto/howto_css_switch.asp converted to tailwind and react
 */
const AwardButton = (props) => {
  const { handleAwardChange, award } = props;

  return (
    <div className="flex flex-row gap-2">
      <label className="switch">
        <input type="checkbox" checked={award} onChange={handleAwardChange}/>
        <span className="slider round"></span>
      </label>
      <p>Awarded?</p>
    </div>
  );
};

AwardButton.propTypes = {
  handleAwardChange: PropTypes.func.isRequired,
  award: PropTypes.bool.isRequired
};

export default AwardButton;
