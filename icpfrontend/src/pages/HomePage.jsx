import PreviewContent from "../components/preview/PreviewContent"
/**
 * 
 * Simple homepage
 * @author Louis Figes
 * @generated GitHub Copilot was used in the creation of this code.
 * 
 */

function HomePage() {

  return (
    <div className="flex flex-col text-center">
      <h1 className="text-6xl font-bold">Welcome to the homepage</h1>
      <p className="text-xl mt-4">For you.</p>
        <PreviewContent />
    </div>
  );
}

export default HomePage;
